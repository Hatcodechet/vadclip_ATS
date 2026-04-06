#!/usr/bin/env python3
"""
Prepare missing videos for a Holmes-VAU/HIVAU-style annotation workflow.

This script follows the paper's pipeline in a practical way:
1. bootstrap: create raw-annotation skeletons from missing MP4 files
2. build-clips: derive clip segments from event boundaries
3. export-prompts: emit prompt tasks for clip captions, event summaries, video summaries
4. apply-responses: merge generated responses back into the annotation JSON

The exported prompts keep the paper's logic but ask for strict JSON responses
for event/video summaries so they can be ingested reliably.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import cv2


CLIP_CAPTION_PROMPT = (
    "Please provide a short and brief description of the video clip, "
    "focusing on the main subjects and their actions. "
    "Describe only what is clearly visible. "
    "Do not infer anomaly type, crime, cause, or intent unless directly visible."
)


EVENT_SUMMARY_PROMPT = """The dense caption of the video is:
{clip_captions}

There {verb} abnormal events ({label_text}) in the video.

Return strict JSON with exactly these keys:
{{
  "judgement": "...",
  "description": "...",
  "analysis": "..."
}}

Requirements:
- Follow the Holmes-VAU event-summary style.
- Use only information supported by the clip captions.
- Do not force the anomaly label if the visible evidence is weak or absent.
- "judgement":
  - If anomaly evidence is clearly visible, state that an anomaly exists and name it.
  - If the event appears normal, state that no anomaly exists in this event.
  - If the captions are insufficient or ambiguous, state that the anomaly is not clearly visible from the available captions.
- "description": summarize the visible event in natural language.
- "analysis": explain the visible basis for the judgement, or explicitly say the evidence is insufficient.
- Do not mention these instructions in the answer.
"""


VIDEO_SUMMARY_PROMPT = """Below is a summary of all the events in the video:
{event_summaries}

There {verb} abnormal events ({label_text}) in the video.

Return strict JSON with exactly these keys:
{{
  "judgement": "...",
  "description": "...",
  "analysis": "..."
}}

Requirements:
- Follow the Holmes-VAU video-summary style.
- Use only information supported by the event summaries.
- Do not force the anomaly label if the evidence is weak or absent.
- "judgement":
  - If the evidence clearly supports an anomaly, state that an anomaly exists and name it.
  - If the video appears normal, state that no anomaly exists in the video.
  - If the evidence is insufficient or ambiguous, say the anomaly is not clearly visible from the available evidence.
- "description": summarize the visible event progression from start to end.
- "analysis": explain the visible basis for the judgement, or explicitly say the evidence is insufficient.
- Do not mention these instructions in the answer.
"""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def strip_json_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        out: list[str] = []
        in_string = False
        escaped = False
        idx = 0
        while idx < len(line):
            ch = line[idx]
            if in_string:
                out.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                idx += 1
                continue
            if ch == '"':
                in_string = True
                out.append(ch)
                idx += 1
                continue
            if ch == "/" and idx + 1 < len(line) and line[idx + 1] == "/":
                break
            out.append(ch)
            idx += 1
        lines.append("".join(out))
    return "\n".join(lines)


def load_json_any(path: Path) -> Any:
    return json.loads(strip_json_comments(read_text(path)))


def dump_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def get_video_info(video_path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or n_frames <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}: fps={fps}, n_frames={n_frames}")
    return n_frames, fps


def stem_to_label(stem: str) -> list[str]:
    if stem.startswith("Normal_Videos_"):
        return []
    match = re.match(r"([A-Za-z_]+?)(\d+_x264)?$", stem)
    if not match:
        return []
    return [match.group(1)]


def build_diff_index(diff_path: Path | None) -> dict[str, dict[str, Any]]:
    if not diff_path:
        return {}
    raw = load_json_any(diff_path)
    if isinstance(raw, dict) and "missing_in_hivau" in raw:
        items = raw["missing_in_hivau"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported diff format: {diff_path}")
    return {item["normalized_name"]: item for item in items}


def build_empty_summary_split() -> dict[str, str]:
    return {"judgement": "", "description": "", "analysis": ""}


def make_entry(stem: str, n_frames: int, fps: float, diff_item: dict[str, Any] | None) -> dict[str, Any]:
    label = stem_to_label(stem)
    if diff_item and diff_item.get("video_name"):
        parent = diff_item["video_name"].split("/")[0]
        label = [] if "Normal_Videos" in parent else [parent]
    rough_description = ""
    if diff_item:
        rough_description = diff_item.get("video_description", "")
    return {
        "n_frames": n_frames,
        "fps": round(fps, 3),
        "label": label,
        "events": [],
        "clips": [],
        "clips_caption": [],
        "events_summary": [],
        "video_summary": rough_description,
        "events_summary_split": [],
        "video_summary_split": build_empty_summary_split(),
    }


def command_bootstrap(args: argparse.Namespace) -> None:
    videos_dir = Path(args.videos_dir)
    output_path = Path(args.output)
    diff_index = build_diff_index(Path(args.diff_json) if args.diff_json else None)
    if output_path.exists():
        data = load_json_any(output_path)
    else:
        data = {}

    added = 0
    updated = 0
    for video_path in sorted(videos_dir.glob("*.mp4")):
        stem = video_path.stem
        n_frames, fps = get_video_info(video_path)
        diff_item = diff_index.get(stem)
        new_entry = make_entry(stem, n_frames, fps, diff_item)
        if stem in data:
            data[stem]["n_frames"] = new_entry["n_frames"]
            data[stem]["fps"] = new_entry["fps"]
            if not data[stem].get("label"):
                data[stem]["label"] = new_entry["label"]
            updated += 1
        else:
            data[stem] = new_entry
            added += 1

    dump_json(output_path, data)
    print(f"Saved {output_path}")
    print(f"Added {added} entries, updated metadata for {updated} entries.")


def build_random_clips_for_event(
    start: float,
    end: float,
    min_len: float,
    max_len: float,
    rng: random.Random,
) -> list[list[float]]:
    if end <= start:
        return []
    clips: list[list[float]] = []
    cursor = start
    while cursor < end:
        remaining = end - cursor
        if remaining <= max_len:
            clips.append([round(cursor, 3), round(end, 3)])
            break
        clip_len = rng.uniform(min_len, max_len)
        clip_end = min(end, cursor + clip_len)
        if clip_end - cursor < min_len and clips:
            clips[-1][1] = round(end, 3)
            break
        clips.append([round(cursor, 3), round(clip_end, 3)])
        cursor = clip_end
    return clips


def ensure_parallel_list_lengths(entry: dict[str, Any]) -> None:
    n_events = len(entry.get("events", []))
    for key in ("clips", "clips_caption", "events_summary", "events_summary_split"):
        value = entry.setdefault(key, [])
        if key in ("events_summary", "events_summary_split"):
            while len(value) < n_events:
                value.append(build_empty_summary_split() if key == "events_summary_split" else "")
        else:
            while len(value) < n_events:
                value.append([])
    entry.setdefault("video_summary", "")
    entry.setdefault("video_summary_split", build_empty_summary_split())


def sample_normal_events(
    duration: float,
    num_events: int,
    min_seconds: float,
    max_seconds: float,
    rng: random.Random,
) -> list[list[float]]:
    if duration <= 0:
        return []
    events: list[list[float]] = []
    attempts = 0
    max_attempts = max(20, num_events * 20)
    max_len = min(max_seconds, duration)
    min_len = min(min_seconds, max_len)
    if min_len <= 0 or max_len <= 0:
        return []

    while len(events) < num_events and attempts < max_attempts:
        attempts += 1
        seg_len = rng.uniform(min_len, max_len)
        if seg_len >= duration:
            start = 0.0
            end = duration
        else:
            start = rng.uniform(0.0, max(0.0, duration - seg_len))
            end = start + seg_len

        overlaps = False
        for cur_start, cur_end in events:
            if not (end <= cur_start or start >= cur_end):
                overlaps = True
                break
        if overlaps:
            continue
        events.append([round(start, 3), round(min(end, duration), 3)])

    events.sort(key=lambda x: x[0])
    if not events:
        events = [[0.0, round(min(duration, max_len), 3)]]
    return events


def command_fill_normal_events(args: argparse.Namespace) -> None:
    annotation_path = Path(args.annotation)
    data = load_json_any(annotation_path)
    rng = random.Random(args.seed)

    touched = 0
    for _, entry in data.items():
        label = entry.get("label", [])
        if label:
            continue
        if entry.get("events") and not args.overwrite:
            continue

        fps = float(entry.get("fps", 0))
        n_frames = int(entry.get("n_frames", 0))
        duration = (n_frames / fps) if fps > 0 else 0.0
        entry["events"] = sample_normal_events(
            duration=duration,
            num_events=args.num_events,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            rng=rng,
        )
        entry["clips"] = []
        entry["clips_caption"] = []
        entry["events_summary"] = []
        entry["events_summary_split"] = []
        entry["video_summary_split"] = build_empty_summary_split()
        touched += 1

    dump_json(annotation_path, data)
    print(f"Filled random normal events for {touched} videos in {annotation_path}")


def command_build_clips(args: argparse.Namespace) -> None:
    annotation_path = Path(args.annotation)
    data = load_json_any(annotation_path)
    rng = random.Random(args.seed)

    touched = 0
    for video_id, entry in data.items():
        events = entry.get("events", [])
        if not events:
            continue
        if entry.get("clips") and not args.overwrite:
            continue
        clips = []
        for event in events:
            start, end = float(event[0]), float(event[1])
            clips.append(build_random_clips_for_event(start, end, args.min_len, args.max_len, rng))
        entry["clips"] = clips
        entry["clips_caption"] = [["" for _ in event_clips] for event_clips in clips]
        ensure_parallel_list_lengths(entry)
        touched += 1
        data[video_id] = entry

    dump_json(annotation_path, data)
    print(f"Updated clips for {touched} videos in {annotation_path}")


def label_text(label: list[str]) -> str:
    return label[0] if label else "None"


def abnormal_verb(label: list[str]) -> str:
    return "are" if label else "is no"


def anomaly_clause(label: list[str]) -> str:
    if label:
        return f"There are abnormal events ({label_text(label)}) in the video."
    return "There is no abnormal event in the video."


def flatten_clip_captions(captions: list[str]) -> str:
    lines = []
    for idx, caption in enumerate(captions, start=1):
        lines.append(f"{idx}. {caption or '[PENDING_CLIP_CAPTION]'}")
    return "\n".join(lines)


def flatten_event_summaries(summaries: list[str]) -> str:
    lines = []
    for idx, summary in enumerate(summaries, start=1):
        lines.append(f"Event {idx}: {summary or '[PENDING_EVENT_SUMMARY]'}")
    return "\n".join(lines)


def command_export_prompts(args: argparse.Namespace) -> None:
    annotation_path = Path(args.annotation)
    output_path = Path(args.output)
    data = load_json_any(annotation_path)

    tasks: list[dict[str, Any]] = []
    for video_id, entry in data.items():
        ensure_parallel_list_lengths(entry)
        events = entry.get("events", [])
        clips = entry.get("clips", [])
        clip_captions = entry.get("clips_caption", [])
        event_summaries = entry.get("events_summary", [])
        label = entry.get("label", [])

        for e_idx, event_clips in enumerate(clips):
            for c_idx, segment in enumerate(event_clips):
                existing_caption = ""
                if e_idx < len(clip_captions) and c_idx < len(clip_captions[e_idx]):
                    existing_caption = clip_captions[e_idx][c_idx]
                tasks.append(
                    {
                        "id": f"{video_id}_E{e_idx}C{c_idx}",
                        "type": "clip_caption",
                        "video_id": video_id,
                        "event_index": e_idx,
                        "clip_index": c_idx,
                        "segment": segment,
                        "prompt": CLIP_CAPTION_PROMPT,
                        "response": "" if args.clear_responses else existing_caption,
                    }
                )

        for e_idx, event in enumerate(events):
            captions = clip_captions[e_idx] if e_idx < len(clip_captions) else []
            existing_summary = event_summaries[e_idx] if e_idx < len(event_summaries) else ""
            tasks.append(
                {
                    "id": f"{video_id}_E{e_idx}",
                    "type": "event_summary",
                    "video_id": video_id,
                    "event_index": e_idx,
                    "segment": event,
                    "label": label,
                    "prompt": EVENT_SUMMARY_PROMPT.format(
                        clip_captions=flatten_clip_captions(captions),
                        verb=abnormal_verb(label),
                        label_text=label_text(label),
                    ),
                    "response": "" if args.clear_responses else existing_summary,
                }
            )

        tasks.append(
            {
                "id": video_id,
                "type": "video_summary",
                "video_id": video_id,
                "label": label,
                "prompt": VIDEO_SUMMARY_PROMPT.format(
                    event_summaries=flatten_event_summaries(event_summaries),
                    verb=abnormal_verb(label),
                    label_text=label_text(label),
                ),
                "response": "" if args.clear_responses else entry.get("video_summary", ""),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Exported {len(tasks)} prompt tasks to {output_path}")


def summary_split_to_text(summary_split: dict[str, str]) -> str:
    parts = [summary_split.get("judgement", ""), summary_split.get("description", ""), summary_split.get("analysis", "")]
    return " ".join(part.strip() for part in parts if part.strip())


def normalize_summary_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def parse_summary_response(raw: str) -> tuple[dict[str, str], str]:
    raw = raw.strip()
    parsed = json.loads(raw)
    summary = {
        "judgement": normalize_summary_text(str(parsed.get("judgement", ""))),
        "description": normalize_summary_text(str(parsed.get("description", ""))),
        "analysis": normalize_summary_text(str(parsed.get("analysis", ""))),
    }
    return summary, summary_split_to_text(summary)


def command_apply_responses(args: argparse.Namespace) -> None:
    annotation_path = Path(args.annotation)
    responses_path = Path(args.responses)
    data = load_json_any(annotation_path)
    applied = 0

    with responses_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            response = item.get("response", "")
            if not response:
                continue

            video_id = item["video_id"]
            if video_id not in data:
                continue
            entry = data[video_id]
            ensure_parallel_list_lengths(entry)

            if item["type"] == "clip_caption":
                e_idx = int(item["event_index"])
                c_idx = int(item["clip_index"])
                while len(entry["clips_caption"]) <= e_idx:
                    entry["clips_caption"].append([])
                while len(entry["clips_caption"][e_idx]) <= c_idx:
                    entry["clips_caption"][e_idx].append("")
                entry["clips_caption"][e_idx][c_idx] = response.strip()
                applied += 1
            elif item["type"] == "event_summary":
                e_idx = int(item["event_index"])
                try:
                    summary_split, merged = parse_summary_response(response)
                except json.JSONDecodeError:
                    continue
                while len(entry["events_summary"]) <= e_idx:
                    entry["events_summary"].append("")
                while len(entry["events_summary_split"]) <= e_idx:
                    entry["events_summary_split"].append(build_empty_summary_split())
                entry["events_summary"][e_idx] = merged
                entry["events_summary_split"][e_idx] = summary_split
                applied += 1
            elif item["type"] == "video_summary":
                try:
                    summary_split, merged = parse_summary_response(response)
                except json.JSONDecodeError:
                    continue
                entry["video_summary"] = merged
                entry["video_summary_split"] = summary_split
                applied += 1

    dump_json(annotation_path, data)
    print(f"Applied {applied} responses into {annotation_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare missing HIVAU annotations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Create or update raw-annotation skeletons from MP4 files.")
    bootstrap.add_argument("--videos-dir", required=True, help="Directory containing missing MP4 files.")
    bootstrap.add_argument("--output", required=True, help="Output annotation JSON path.")
    bootstrap.add_argument("--diff-json", help="Optional diff JSON/JSONC used to infer labels and source metadata.")
    bootstrap.set_defaults(func=command_bootstrap)

    build_clips = subparsers.add_parser("build-clips", help="Generate clip segments from event boundaries.")
    build_clips.add_argument("--annotation", required=True, help="Annotation JSON path.")
    build_clips.add_argument("--min-len", type=float, default=3.0, help="Minimum clip length in seconds.")
    build_clips.add_argument("--max-len", type=float, default=8.0, help="Maximum clip length in seconds.")
    build_clips.add_argument("--seed", type=int, default=42, help="Random seed.")
    build_clips.add_argument("--overwrite", action="store_true", help="Overwrite existing clips.")
    build_clips.set_defaults(func=command_build_clips)

    fill_normal_events = subparsers.add_parser(
        "fill-normal-events",
        help="Randomly crop normal event segments for videos with empty labels.",
    )
    fill_normal_events.add_argument("--annotation", required=True, help="Annotation JSON path.")
    fill_normal_events.add_argument("--num-events", type=int, default=2, help="Number of random normal events per video.")
    fill_normal_events.add_argument("--min-seconds", type=float, default=8.0, help="Minimum event duration in seconds.")
    fill_normal_events.add_argument("--max-seconds", type=float, default=25.0, help="Maximum event duration in seconds.")
    fill_normal_events.add_argument("--seed", type=int, default=42, help="Random seed.")
    fill_normal_events.add_argument("--overwrite", action="store_true", help="Overwrite existing normal events.")
    fill_normal_events.set_defaults(func=command_fill_normal_events)

    export_prompts = subparsers.add_parser("export-prompts", help="Export prompt tasks for captioning and summarization.")
    export_prompts.add_argument("--annotation", required=True, help="Annotation JSON path.")
    export_prompts.add_argument("--output", required=True, help="Output JSONL path.")
    export_prompts.add_argument(
        "--clear-responses",
        action="store_true",
        help="Reset all response fields to empty when exporting prompts.",
    )
    export_prompts.set_defaults(func=command_export_prompts)

    apply_responses = subparsers.add_parser("apply-responses", help="Apply clip/event/video responses back to annotations.")
    apply_responses.add_argument("--annotation", required=True, help="Annotation JSON path.")
    apply_responses.add_argument("--responses", required=True, help="Responses JSONL path.")
    apply_responses.set_defaults(func=command_apply_responses)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
