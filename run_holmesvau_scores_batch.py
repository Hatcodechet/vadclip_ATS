#!/venv/holmesvau/bin/python
"""
Batch Holmes-VAU inference from precomputed VadCLIP temporal scores.

Pipeline:
    video + fused.npy -> ATS (score-driven) -> Holmes-VAU description

This script scans /workspace/outputs for fused.npy files, matches them to videos
under /workspace/test by basename, randomly picks prompts from the existing
HolmesVAU prompt pools, and writes one consolidated JSON file containing
video_name and video_description for every processed video.
"""

import argparse
import glob
import json
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HOLMESVAU_DIR = "/workspace/score_fuse/HolmesVAU"
sys.path.insert(0, HOLMESVAU_DIR)
os.chdir(HOLMESVAU_DIR)

from holmesvau.holmesvau_utils import load_model, generate


PROMPT_LIST = [
    "Describe the anomaly events observed in the video.",
    "Could you describe the anomaly events observed in the video?",
    "Could you specify the anomaly events present in the video?",
    "Give a description of the detected anomaly events in this video.",
    "Could you give a description of the anomaly events in the video?",
    "Provide a summary of the anomaly events in the video.",
    "Could you provide a summary of the anomaly events in this video?",
    "What details can you provide about the anomaly in the video?",
    "How would you detail the anomaly events found in the video?",
    "How would you describe the particular anomaly events in the video?",
]

NORMAL_PROMPT_LIST = [
    "Describe the events occurring in this video.",
    "Could you describe the events observed in the video?",
    "Provide a detailed description of the actions in this video.",
    "Give a summary of what is happening in this video.",
    "Could you specify the actions and subjects present in the video?",
    "What details can you provide about the events in the video?",
    "How would you describe the scene in this video?",
    "Provide a video-level description of this video.",
    "Summarize the main subjects and key actions in this video.",
    "Describe the main activities unfolding in the video.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Holmes-VAU description generation from fused.npy scores."
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="/workspace/outputs",
        help="Directory containing per-video fused.npy outputs.",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="/workspace/test",
        help="Root directory containing video files.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/score_fuse/HolmesVAU/ckpts/HolmesVAU-2B",
        help="Path to HolmesVAU-2B checkpoint directory.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/workspace/outputs/holmesvau_video_descriptions_290.json",
        help="Path to the final JSON output file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--select_frames",
        type=int,
        default=12,
        help="Number of frames passed to Holmes-VAU.",
    )
    parser.add_argument(
        "--dense_sample_freq",
        type=int,
        default=16,
        help="Dense frame step used for score-to-frame mapping.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Density-aware ATS tau.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for prompt selection.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save progress every N successful videos.",
    )
    return parser.parse_args()


def build_video_map(video_root):
    video_map = {}
    pattern = os.path.join(video_root, "**", "*.mp4")
    for video_path in glob.glob(pattern, recursive=True):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if video_name not in video_map:
            video_map[video_name] = video_path
    return video_map


def find_score_jobs(outputs_dir):
    jobs = []
    for score_path in sorted(glob.glob(os.path.join(outputs_dir, "*", "fused.npy"))):
        video_name = os.path.basename(os.path.dirname(score_path))
        jobs.append(
            {
                "video_name": video_name,
                "score_path": score_path,
            }
        )
    return jobs


def is_normal_video(video_name, video_path):
    if video_name.startswith("Normal_"):
        return True
    return "Testing_Normal_Videos_Anomaly" in video_path


def load_existing_results(output_json):
    output_path = Path(output_json)
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {item["video_name"]: item for item in data}
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unsupported JSON structure in {output_json}.")


def save_results(output_json, results_by_name):
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_items = [results_by_name[name] for name in sorted(results_by_name)]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ordered_items, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    random.seed(args.seed)

    print("=" * 80)
    print("Holmes-VAU Batch Description Generation From Scores")
    print("=" * 80)
    print(f"Outputs dir : {args.outputs_dir}")
    print(f"Video root  : {args.video_root}")
    print(f"Model path  : {args.model_path}")
    print(f"Output json : {args.output_json}")
    print(f"Device      : {args.device}")

    if not os.path.isfile(os.path.join(args.model_path, "model.safetensors")):
        raise FileNotFoundError(
            f"HolmesVAU-2B checkpoint not found at {args.model_path}"
        )

    video_map = build_video_map(args.video_root)
    jobs = find_score_jobs(args.outputs_dir)
    existing = load_existing_results(args.output_json)

    print(f"Indexed videos        : {len(video_map)}")
    print(f"Found fused.npy jobs  : {len(jobs)}")
    print(f"Existing JSON entries : {len(existing)}")

    device = torch.device(args.device)
    model, tokenizer, generation_config, sampler = load_model(
        args.model_path,
        sampler_path=None,
        device=device,
        sampler_tau=args.tau,
    )
    generation_config["do_sample"] = False
    generation_config["max_new_tokens"] = args.max_new_tokens

    results_by_name = dict(existing)
    processed = 0
    success = 0
    missing_video = []
    failed = []

    for idx, job in enumerate(jobs, start=1):
        video_name = job["video_name"]
        score_path = job["score_path"]

        if video_name in results_by_name:
            print(f"[{idx}/{len(jobs)}] {video_name} already exists in JSON, skipping")
            continue

        video_path = video_map.get(video_name)
        if video_path is None:
            print(f"[{idx}/{len(jobs)}] {video_name} missing matching video file")
            missing_video.append(video_name)
            continue

        is_normal = is_normal_video(video_name, video_path)
        prompt_pool = NORMAL_PROMPT_LIST if is_normal else PROMPT_LIST
        prompt = random.choice(prompt_pool)
        scores = np.load(score_path)

        print(f"[{idx}/{len(jobs)}] {video_name}")
        print(f"  video : {video_path}")
        print(f"  score : {score_path}")
        print(f"  prompt: {prompt}")

        try:
            pred, history, frame_indices, anomaly_score = generate(
                video_path=video_path,
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                sampler=sampler,
                dense_sample_freq=args.dense_sample_freq,
                select_frames=args.select_frames,
                use_ATS=True,
                external_scores=scores,
            )
            description = pred.strip()
            results_by_name[video_name] = {
                "video_name": video_name,
                "video_description": description,
            }
            success += 1
            processed += 1
            print(f"  done  : {description[:160]}")

            if success % args.save_every == 0:
                save_results(args.output_json, results_by_name)

        except Exception as exc:
            processed += 1
            failed.append(video_name)
            print(f"  error : {exc}")
            traceback.print_exc()

    save_results(args.output_json, results_by_name)

    print("\n" + "=" * 80)
    print("Final Report")
    print("=" * 80)
    print(f"Total jobs              : {len(jobs)}")
    print(f"Successful descriptions : {success}")
    print(f"Missing videos          : {len(missing_video)}")
    print(f"Failed videos           : {len(failed)}")
    print(f"JSON saved to           : {args.output_json}")

    if missing_video:
        print("\nMissing video names:")
        print(", ".join(missing_video))

    if failed:
        print("\nFailed video names:")
        print(", ".join(failed))


if __name__ == "__main__":
    main()
