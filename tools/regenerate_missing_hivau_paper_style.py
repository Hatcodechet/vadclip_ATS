#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from decord import VideoReader, cpu
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaNextVideoForConditionalGeneration,
)


CLIP_CAPTION_PROMPT = (
    "Please provide a short and brief description of the video clip, focusing on the main subjects and their actions."
)

EVIDENCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "Abuse": ("abuse", "beat", "beating", "kick", "kicking", "slap", "slapped", "punch", "punched", "strike", "drag", "dragged", "hit", "hitting"),
    "Assault": ("assault", "attack", "attacking", "punch", "punched", "kick", "kicking", "strike", "striking", "bat", "stick", "weapon", "hit", "hitting", "confrontation", "altercation"),
    "Explosion": ("explosion", "explode", "exploded", "blast", "fireball", "flame", "flames", "smoke", "burning", "burn", "fire"),
    "RoadAccidents": ("accident", "crash", "collision", "collide", "collided", "wreck", "hit", "impact", "debris"),
    "Shooting": ("shooting", "shoot", "shot", "gun", "gunfire", "firearm", "rifle", "pistol", "muzzle", "armed", "weapon"),
    "Shoplifting": ("shoplifting", "shoplift", "steal", "stole", "stolen", "theft", "conceal", "concealed", "pocket", "hide", "hidden", "bag"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def dump_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \"'")


def normalize_caption(text: str, max_words: int = 72) -> str:
    text = clean_text(text)
    sentence_match = re.search(r"^(.+?[.!?])(\s|$)", text)
    if sentence_match:
        text = sentence_match.group(1).strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:")
    if text and text[-1] not in ".!?":
        text += "."
    return text


def segment_to_frame_bounds(vr: VideoReader, segment: list[float]) -> tuple[int, int]:
    fps = float(vr.get_avg_fps())
    start_idx = max(0, min(len(vr) - 1, int(round(float(segment[0]) * fps))))
    end_idx = max(start_idx + 1, min(len(vr), int(round(float(segment[1]) * fps))))
    return start_idx, end_idx


def sample_uniform_indices(vr: VideoReader, segment: list[float], num_frames: int) -> list[int]:
    start_idx, end_idx = segment_to_frame_bounds(vr, segment)
    span = end_idx - start_idx
    actual = min(num_frames, max(2, span))
    if actual <= 1:
        return [start_idx]
    return torch.linspace(start_idx, end_idx - 1, steps=actual).round().to(torch.int64).tolist()


def dedupe_keep_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def sample_frames_by_indices(video_path: Path, indices: list[int]) -> list[Any]:
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    return [vr[int(i)].asnumpy() for i in indices]


def compute_ats_segment_indices(
    video_path: Path,
    segment: list[float],
    num_frames: int,
    ats_bundle: dict[str, Any],
    dense_sample_freq: int = 16,
) -> list[int]:
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    uniform_indices = sample_uniform_indices(vr, segment, num_frames)
    start_idx, end_idx = segment_to_frame_bounds(vr, segment)
    dense_indices = list(range(start_idx, end_idx, max(1, dense_sample_freq)))
    if len(dense_indices) < max(2, num_frames):
        return uniform_indices
    pixel_values, _ = ats_bundle["get_pixel_values"](vr, dense_indices)
    _, sampled_local = ats_bundle["sampler"].density_aware_sample(
        pixel_values,
        ats_bundle["model"],
        select_frames=min(num_frames, len(dense_indices)),
    )
    indices = dedupe_keep_order([dense_indices[int(i)] for i in sampled_local])
    if len(indices) < num_frames:
        indices = dedupe_keep_order(indices + uniform_indices)
    return indices[:num_frames]


def sample_segment_frames(
    video_path: Path,
    segment: list[float],
    num_frames: int,
    selected_indices: list[int] | None = None,
    ats_bundle: dict[str, Any] | None = None,
    dense_sample_freq: int = 16,
) -> list[Any]:
    if selected_indices is not None:
        return sample_frames_by_indices(video_path, selected_indices)
    if ats_bundle is not None:
        indices = compute_ats_segment_indices(
            video_path,
            segment,
            num_frames=num_frames,
            ats_bundle=ats_bundle,
            dense_sample_freq=dense_sample_freq,
        )
        return sample_frames_by_indices(video_path, indices)
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    indices = sample_uniform_indices(vr, segment, num_frames)
    return [vr[int(i)].asnumpy() for i in indices]


def build_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_clip_model(model_id: str):
    quant_config = build_quant_config()
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return processor, model


def load_holmes_ats_bundle(holmes_root: Path, holmes_model_path: Path, sampler_path: Path, device: str) -> dict[str, Any]:
    holmes_root_str = str(holmes_root)
    if holmes_root_str not in sys.path:
        sys.path.insert(0, holmes_root_str)
    from holmesvau.holmesvau_utils import get_pixel_values, load_model  # type: ignore

    model, _, _, sampler = load_model(str(holmes_model_path), str(sampler_path), torch.device(device))
    return {
        "model": model,
        "sampler": sampler,
        "get_pixel_values": get_pixel_values,
    }


def make_video_prompt() -> str:
    return f"USER: <video>\n{CLIP_CAPTION_PROMPT}\nASSISTANT:"


@torch.inference_mode()
def generate_clip_caption(
    processor,
    model,
    video_path: Path,
    segment: list[float],
    num_frames: int,
    selected_indices: list[int] | None = None,
    ats_bundle: dict[str, Any] | None = None,
    dense_sample_freq: int = 16,
) -> str:
    clip = sample_segment_frames(
        video_path,
        segment,
        num_frames=num_frames,
        selected_indices=selected_indices,
        ats_bundle=ats_bundle,
        dense_sample_freq=dense_sample_freq,
    )
    prompt = make_video_prompt()
    inputs = processor(text=prompt, videos=[clip], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
    )
    decoded = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    if "ASSISTANT:" in decoded:
        decoded = decoded.split("ASSISTANT:", 1)[1]
    elif "ASSISTANT" in decoded:
        decoded = decoded.split("ASSISTANT", 1)[-1]
    return normalize_caption(decoded)


def unload_model(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_text_model(model_id: str):
    quant_config = build_quant_config()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def label_text(label: list[str]) -> str:
    return label[0] if label else "None"


def abnormal_sentence(label: list[str]) -> str:
    return f"There are abnormal events ({label_text(label)}) in the video." if label else "There is no abnormal events (None) in the video."


def build_event_prompt(captions: list[str], label: list[str]) -> str:
    dense = "\n".join(f"{idx + 1}. {cap}" for idx, cap in enumerate(captions))
    return (
        f"The dense caption of the video is:\n{dense}\n\n"
        f"{abnormal_sentence(label)}\n"
        "Your response should include:\n"
        "1. Whether the anomaly exists and the specific name of the anomaly.\n"
        "2. A summary of the anomaly events.\n"
        "3. Brief explanation of the basis for judging the anomaly.\n\n"
        "Write in the HolmesVAU raw-annotation style.\n"
        "Use these sentence patterns when appropriate:\n"
        '- judgement: "An anomaly exists, specifically an \\"<LABEL>\\" anomaly." or "There is no anomaly in the video."\n'
        '- description: "The anomaly event involves ..." or "The video depicts normal activities ..."\n'
        '- analysis: "The basis for judging this anomaly is ..." or "The basis for judging that no anomaly exists is ..."\n\n'
        "Important constraints:\n"
        "- Use only entities, actions, and evidence supported by the dense captions.\n"
        "- Do not invent sounds, reactions, explosions, gunshots, panic, theft, or injuries unless they are explicitly supported by the captions.\n"
        "- The label provides category context, but it is not by itself evidence.\n"
        "- If the captions do not support the labeled anomaly, say there is no clear anomaly in the video.\n\n"
        "Return strict JSON with keys:\n"
        "judgement, description, analysis."
    )


def build_video_prompt(event_summaries: list[str], label: list[str]) -> str:
    dense = "\n".join(f"Event {idx + 1}: {summary}" for idx, summary in enumerate(event_summaries))
    return (
        f"Below is a summary of all the events in the video:\n{dense}\n\n"
        f"{abnormal_sentence(label)}\n"
        "Your response should include:\n"
        "1. Whether the anomaly exists and the specific name of the anomaly.\n"
        "2. Detailed description of the video anomaly event from start to end.\n"
        "3. Brief analysis of the basis for judging the anomaly.\n\n"
        "Write in the HolmesVAU raw-annotation style.\n"
        "Use these sentence patterns when appropriate:\n"
        '- judgement: "An anomaly exists, specifically an \\"<LABEL>\\" anomaly." or "There is no anomaly in the video."\n'
        '- description: "The anomaly event involves ..." or "The video depicts normal activities ..."\n'
        '- analysis: "The basis for judging this anomaly is ..." or "The basis for judging that no anomaly exists is ..."\n\n'
        "Important constraints:\n"
        "- Use only evidence supported by the event summaries.\n"
        "- Do not add sounds, reactions, explosions, gunshots, panic, theft, or injuries unless they are already supported by the event summaries.\n"
        "- The label provides category context, but it is not by itself evidence.\n"
        "- If the summaries do not support the labeled anomaly, say there is no clear anomaly in the video.\n\n"
        "Return strict JSON with keys:\n"
        "judgement, description, analysis."
    )


def parse_json_payload(text: str) -> dict[str, str] | None:
    text = clean_text(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return normalize_summary_payload(
        {
            "judgement": clean_text(str(data.get("judgement", ""))),
            "description": clean_text(str(data.get("description", ""))),
            "analysis": clean_text(str(data.get("analysis", ""))),
        }
    )


def normalize_judgement_text(text: str, label: str) -> str:
    raw = clean_text(text)
    low = raw.lower()
    if not raw:
        return f'There is no anomaly in the video.' if not label or label == "None" else f'An anomaly exists, specifically an "{label}" anomaly.'
    if "no anomaly" in low or "none" == low or "no clear anomaly" in low or "insufficient" in low or low == "false":
        return "There is no anomaly in the video."
    if "anomaly exists" in low and label and label != "None":
        return f'An anomaly exists, specifically an "{label}" anomaly.'
    if raw in {"Yes", "True"} and label and label != "None":
        return f'An anomaly exists, specifically an "{label}" anomaly.'
    if raw == label and label and label != "None":
        return f'An anomaly exists, specifically an "{label}" anomaly.'
    if label and label != "None" and label.lower() in low:
        return f'An anomaly exists, specifically an "{label}" anomaly.'
    return raw if raw.endswith(".") else raw + "."


def normalize_summary_payload(payload: dict[str, str], label: str | None = None) -> dict[str, str]:
    label_name = label or ""
    judgement = normalize_judgement_text(payload.get("judgement", ""), label_name)
    description = clean_text(payload.get("description", ""))
    analysis = clean_text(payload.get("analysis", ""))
    if description and description[-1] not in ".!?":
        description += "."
    if analysis and analysis[-1] not in ".!?":
        analysis += "."
    low_judgement = judgement.lower()
    if "there is no anomaly in the video" in low_judgement:
        if description and not description.lower().startswith("the video depicts normal activities") and not description.lower().startswith("the video shows normal activities"):
            description = f"The video depicts normal activities. {description}"
        if analysis and not analysis.lower().startswith("the basis for judging that no anomaly exists is"):
            analysis = f"The basis for judging that no anomaly exists is that {analysis[0].lower() + analysis[1:] if len(analysis) > 1 else analysis.lower()}"
    else:
        if description and not description.lower().startswith("the anomaly event involves"):
            description = f"The anomaly event involves {description[0].lower() + description[1:] if len(description) > 1 else description.lower()}"
        if analysis and not analysis.lower().startswith("the basis for judging this anomaly is"):
            analysis = f"The basis for judging this anomaly is that {analysis[0].lower() + analysis[1:] if len(analysis) > 1 else analysis.lower()}"
    return {
        "judgement": judgement,
        "description": description,
        "analysis": analysis,
    }


def evidence_supported(texts: list[str], label: str) -> bool:
    if not label or label == "None":
        return False
    patterns = EVIDENCE_PATTERNS.get(label, ())
    if not patterns:
        return False
    haystack = " ".join(clean_text(text).lower() for text in texts)
    return any(re.search(rf"\b{re.escape(pattern)}\b", haystack) for pattern in patterns)


def build_no_anomaly_payload_from_texts(texts: list[str]) -> dict[str, str]:
    evidence = " ".join(clean_text(text) for text in texts if clean_text(text))
    if not evidence:
        evidence = "The video depicts normal activities."
    description = f"The video depicts normal activities. {evidence}"
    analysis = "The basis for judging that no anomaly exists is that the available descriptions do not provide clear visual evidence of an abnormal event."
    return normalize_summary_payload(
        {
            "judgement": "There is no anomaly in the video.",
            "description": description,
            "analysis": analysis,
        },
        "None",
    )


@torch.inference_mode()
def generate_json_response(
    tokenizer,
    model,
    prompt: str,
    label: list[str] | None = None,
    evidence_texts: list[str] | None = None,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an annotation assistant for video anomaly understanding. "
                "Your writing style must match HolmesVAU raw annotations: explicit anomaly judgement, event-focused description, and basis-for-judgement analysis. "
                "Return strict JSON only, with keys judgement, description, analysis."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    last_decoded = ""
    for max_new_tokens in (256, 384, 512):
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        last_decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        payload = parse_json_payload(last_decoded)
        if payload is not None:
            payload = normalize_summary_payload(payload, label_text(label or []))
            label_name = label_text(label or [])
            if evidence_texts and label_name != "None" and not evidence_supported(evidence_texts, label_name):
                payload = build_no_anomaly_payload_from_texts(evidence_texts)
            return json.dumps(payload, ensure_ascii=False)
    raise RuntimeError(f"Failed to parse JSON response: {last_decoded}")


def summary_text_from_json(raw: str) -> str:
    data = json.loads(raw)
    judgement = clean_text(str(data.get("judgement", ""))).lower()
    description = clean_text(str(data.get("description", "")))
    analysis = clean_text(str(data.get("analysis", "")))
    if "there is no anomaly in the video" in judgement:
        description = re.sub(r"^The video depicts normal activities\.?\s*", "", description, flags=re.IGNORECASE)
        return description
    return " ".join(part for part in (description, analysis) if part)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--clip-model", default="llava-hf/LLaVA-NeXT-Video-7B-hf")
    parser.add_argument("--text-model", default="NousResearch/Hermes-3-Llama-3.1-8B")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--dense-sample-freq", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--skip-clips", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--disable-ats", action="store_true")
    parser.add_argument("--holmes-root", default=str(Path(__file__).resolve().parents[1] / "HolmesVAU"))
    parser.add_argument("--holmes-model", default=str(Path(__file__).resolve().parents[1] / "HolmesVAU" / "ckpts" / "HolmesVAU-2B"))
    parser.add_argument("--sampler-path", default=str(Path(__file__).resolve().parents[1] / "HolmesVAU" / "holmesvau" / "ATS" / "anomaly_scorer.pth"))
    parser.add_argument("--holmes-device", default="cuda:0")
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    videos_dir = Path(args.videos_dir)
    items = load_jsonl(prompts_path)

    clip_items = [item for item in items if item["type"] == "clip_caption"]
    event_items = [item for item in items if item["type"] == "event_summary"]
    video_items = [item for item in items if item["type"] == "video_summary"]

    clip_map: dict[tuple[str, int, int], str] = {}
    if args.skip_clips:
        for item in clip_items:
            clip_map[(item["video_id"], int(item["event_index"]), int(item["clip_index"]))] = item["response"]
    else:
        ats_index_map: dict[tuple[str, int, int], list[int]] = {}
        if not args.disable_ats:
            ats_bundle = load_holmes_ats_bundle(
                Path(args.holmes_root),
                Path(args.holmes_model),
                Path(args.sampler_path),
                args.holmes_device,
            )
            for idx, item in enumerate(clip_items, start=1):
                video_path = videos_dir / f"{item['video_id']}.mp4"
                key = (item["video_id"], int(item["event_index"]), int(item["clip_index"]))
                ats_index_map[key] = compute_ats_segment_indices(
                    video_path,
                    item["segment"],
                    num_frames=args.num_frames,
                    ats_bundle=ats_bundle,
                    dense_sample_freq=args.dense_sample_freq,
                )
                if idx % args.save_every == 0:
                    print(f"[ats] prepared {idx}/{len(clip_items)}")
            print(f"[ats] prepared {len(clip_items)}/{len(clip_items)}")
            unload_model(ats_bundle["model"], ats_bundle["sampler"])
        processor, clip_model = load_clip_model(args.clip_model)
        for idx, item in enumerate(clip_items, start=1):
            video_path = videos_dir / f"{item['video_id']}.mp4"
            caption = generate_clip_caption(
                processor,
                clip_model,
                video_path,
                item["segment"],
                num_frames=args.num_frames,
                selected_indices=ats_index_map.get((item["video_id"], int(item["event_index"]), int(item["clip_index"]))),
                dense_sample_freq=args.dense_sample_freq,
            )
            item["response"] = caption
            clip_map[(item["video_id"], int(item["event_index"]), int(item["clip_index"]))] = caption
            if idx % args.save_every == 0:
                dump_jsonl(prompts_path, items)
                print(f"[clip] regenerated {idx}/{len(clip_items)}")
        dump_jsonl(prompts_path, items)
        print(f"[clip] regenerated {len(clip_items)}/{len(clip_items)}")
        unload_model(processor, clip_model)

    if not args.skip_text:
        tokenizer, text_model = load_text_model(args.text_model)
        event_map: dict[tuple[str, int], str] = {}

        for idx, item in enumerate(event_items, start=1):
            captions: list[str] = []
            clip_idx = 0
            while (item["video_id"], int(item["event_index"]), clip_idx) in clip_map:
                captions.append(clip_map[(item["video_id"], int(item["event_index"]), clip_idx)])
                clip_idx += 1
            raw = generate_json_response(
                tokenizer,
                text_model,
                build_event_prompt(captions, item.get("label", [])),
                item.get("label", []),
                captions,
            )
            item["response"] = raw
            event_map[(item["video_id"], int(item["event_index"]))] = raw
            if idx % args.save_every == 0:
                dump_jsonl(prompts_path, items)
                print(f"[event] regenerated {idx}/{len(event_items)}")

        for idx, item in enumerate(video_items, start=1):
            event_summaries: list[str] = []
            event_idx = 0
            while (item["video_id"], event_idx) in event_map:
                event_summaries.append(summary_text_from_json(event_map[(item["video_id"], event_idx)]))
                event_idx += 1
            raw = generate_json_response(
                tokenizer,
                text_model,
                build_video_prompt(event_summaries, item.get("label", [])),
                item.get("label", []),
                event_summaries,
            )
            item["response"] = raw
            if idx % args.save_every == 0:
                dump_jsonl(prompts_path, items)
                print(f"[video] regenerated {idx}/{len(video_items)}")

        dump_jsonl(prompts_path, items)
        print(f"[event] regenerated {len(event_items)}/{len(event_items)}")
        print(f"[video] regenerated {len(video_items)}/{len(video_items)}")


if __name__ == "__main__":
    main()
