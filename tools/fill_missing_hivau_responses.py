#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from decord import VideoReader, cpu


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def dump_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def label_text(label: list[str]) -> str:
    return label[0] if label else "None"


def judgement_text(label: list[str], level: str) -> str:
    if label:
        return f"An anomaly exists in this {level}, specifically {label_text(label)}."
    return f"No anomaly exists in this {level}."


def clean_text(text: str) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \"'")


def dedupe_sentences(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = clean_text(item).rstrip(".")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def summarize_caption_list(captions: list[str]) -> str:
    uniq = dedupe_sentences(captions)
    if not uniq:
        return ""
    if len(uniq) == 1:
        return uniq[0] + "."
    if len(uniq) == 2:
        return f"{uniq[0]}. Then {uniq[1].lower()}."
    joined = ". Then ".join([uniq[0], *[u.lower() for u in uniq[1:3]]])
    return joined + "."


def build_summary_response(label: list[str], source_sentences: list[str], level: str) -> str:
    description = summarize_caption_list(source_sentences)
    if not description:
        if label:
            description = f"The {level} contains {label_text(label).lower()} activity."
        else:
            description = f"The {level} shows routine activity without a clear anomaly."

    if label:
        analysis = (
            f"This judgement is based on the generated captions describing {description.rstrip('.').lower()} "
            f"which is consistent with {label_text(label)}."
        )
    else:
        analysis = (
            f"This judgement is based on the generated captions describing ordinary activity and no clear "
            f"violent, dangerous, or suspicious event."
        )

    payload = {
        "judgement": judgement_text(label, level),
        "description": description,
        "analysis": clean_text(analysis),
    }
    return json.dumps(payload, ensure_ascii=False)


def load_holmes(repo_dir: Path, model_dir: Path):
    import sys

    sys.path.insert(0, str(repo_dir))
    from holmesvau.holmesvau_utils import get_pixel_values, load_model
    from holmesvau.internvl_utils import get_index

    model, tokenizer, generation_config, _sampler = load_model(
        str(model_dir),
        str(repo_dir / "holmesvau" / "ATS" / "anomaly_scorer.pth"),
        torch.device("cuda:0"),
    )
    generation_config = dict(generation_config)
    generation_config["max_new_tokens"] = 96
    return model, tokenizer, generation_config, get_pixel_values, get_index


def caption_segment(
    video_path: Path,
    segment: list[float],
    prompt: str,
    model,
    tokenizer,
    generation_config: dict,
    get_pixel_values,
    get_index,
    select_frames: int = 8,
) -> str:
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    max_frame = max(0, len(vr) - 1)
    fps = float(vr.get_avg_fps())
    frame_indices = get_index(
        bound=(float(segment[0]), float(segment[1])),
        fps=fps,
        max_frame=max_frame,
        first_idx=0,
        num_segments=select_frames,
    )
    frame_indices = [max(0, min(max_frame, int(i))) for i in frame_indices]
    pixel_values, num_patches_list = get_pixel_values(vr, frame_indices)
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
    video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
    response, _history = model.chat(
        tokenizer,
        pixel_values,
        video_prefix + prompt,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=True,
    )
    torch.cuda.empty_cache()
    return clean_text(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    videos_dir = Path(args.videos_dir)
    repo_dir = Path(args.repo_dir)
    model_dir = Path(args.model_dir)

    items = load_jsonl(prompts_path)
    clip_map: dict[tuple[str, int, int], str] = {}
    event_map: dict[tuple[str, int], str] = {}

    for item in items:
        if item["type"] == "clip_caption" and item.get("response"):
            clip_map[(item["video_id"], int(item["event_index"]), int(item["clip_index"]))] = item["response"]
        elif item["type"] == "event_summary" and item.get("response", "").strip().startswith("{"):
            event_map[(item["video_id"], int(item["event_index"]))] = item["response"]

    model = tokenizer = generation_config = get_pixel_values = get_index = None
    filled = 0
    processed_clip = 0

    for idx, item in enumerate(items):
        if args.limit and filled >= args.limit:
            break

        if item["type"] == "clip_caption" and not item.get("response"):
            if model is None:
                model, tokenizer, generation_config, get_pixel_values, get_index = load_holmes(repo_dir, model_dir)
            video_path = videos_dir / f"{item['video_id']}.mp4"
            caption = caption_segment(
                video_path=video_path,
                segment=item["segment"],
                prompt=item["prompt"],
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                get_pixel_values=get_pixel_values,
                get_index=get_index,
            )
            item["response"] = caption
            clip_map[(item["video_id"], int(item["event_index"]), int(item["clip_index"]))] = caption
            filled += 1
            processed_clip += 1
            print(f"[clip] {filled} filled: {item['id']} -> {caption}")
            if processed_clip % args.save_every == 0:
                dump_jsonl(prompts_path, items)
                print(f"[save] wrote progress after {processed_clip} clip captions")

    for item in items:
        if item["type"] == "event_summary":
            key = (item["video_id"], int(item["event_index"]))
            captions: list[str] = []
            clip_idx = 0
            while (item["video_id"], int(item["event_index"]), clip_idx) in clip_map:
                captions.append(clip_map[(item["video_id"], int(item["event_index"]), clip_idx)])
                clip_idx += 1
            item["response"] = build_summary_response(item.get("label", []), captions, "event")
            event_map[key] = item["response"]

    for item in items:
        if item["type"] != "video_summary":
            continue
        summaries: list[str] = []
        event_idx = 0
        while (item["video_id"], event_idx) in event_map:
            event_payload = json.loads(event_map[(item["video_id"], event_idx)])
            summaries.append(str(event_payload.get("description", "")))
            event_idx += 1
        item["response"] = build_summary_response(item.get("label", []), summaries, "video")

    dump_jsonl(prompts_path, items)
    print(f"[done] wrote {len(items)} tasks to {prompts_path}")


if __name__ == "__main__":
    main()
