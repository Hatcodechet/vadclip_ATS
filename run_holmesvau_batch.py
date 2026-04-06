#!/venv/main/bin/python3
"""
Batch HolmesVAU inference for ucf_database_test.json.
Processes entries where video_summary == "".
Uses HolmesVAU-2B model + anomaly scorer, consistent with inference.py.
"""

import os
import sys
import json
import glob
import traceback
import random

# Add HolmesVAU to path
HOLMESVAU_DIR = '/workspace/Capstone/HolmesVAU'
sys.path.insert(0, HOLMESVAU_DIR)
os.chdir(HOLMESVAU_DIR)  # needed for relative imports in holmesvau package

import torch
from holmesvau.holmesvau_utils import load_model, generate

# ── Paths ──────────────────────────────────────────────────────────────────
MLLM_PATH    = '/workspace/Capstone/HolmesVAU/ckpts/HolmesVAU-2B'
SAMPLER_PATH = '/workspace/Capstone/HolmesVAU/holmesvau/ATS/anomaly_scorer.pth'
JSON_PATH    = '/workspace/Capstone/HIVAU-70k-NEW/ucf_database_test.json'

VIDEO_SEARCH_DIRS = [
    '/workspace/Testing_Anomaly_Videos',
    '/workspace/Testing_Normal_Videos_Anomaly',
    '/workspace/Capstone/Missing',
]

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
    "How would you describe the particular anomaly events in the video?"
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
    "Describe the main activities unfolding in the video."
]

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_video_map(search_dirs):
    """Build a dict: video_id -> absolute path"""
    video_map = {}
    for search_dir in search_dirs:
        for mp4 in glob.glob(os.path.join(search_dir, '**', '*.mp4'), recursive=True):
            vid_id = os.path.splitext(os.path.basename(mp4))[0]
            if vid_id not in video_map:
                video_map[vid_id] = mp4
    return video_map


def main():
    print("=" * 60)
    print("HolmesVAU Batch Inference")
    print("=" * 60)

    # ── Checkpoint check ───────────────────────────────────────────────────
    checkpoint_downloaded = False
    if not os.path.isfile(os.path.join(MLLM_PATH, 'model.safetensors')):
        print(f"HolmesVAU-2B checkpoint NOT found at {MLLM_PATH}")
        print("Downloading from: https://huggingface.co/ppxin321/HolmesVAU-2B")
        os.makedirs(MLLM_PATH, exist_ok=True)
        ret = os.system(
            f"huggingface-cli download ppxin321/HolmesVAU-2B --local-dir {MLLM_PATH}"
        )
        if ret != 0:
            print("ERROR: Download failed.")
            sys.exit(1)
        checkpoint_downloaded = True
        print("Download complete.")
    else:
        print(f"HolmesVAU-2B checkpoint found at {MLLM_PATH}")

    # ── Load target JSON ───────────────────────────────────────────────────
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # Process ALL entries as requested by the user
    targets = list(data.keys())
    print(f"\nTotal entries: {len(data)}")
    print(f"Entries to process (overwrite all): {len(targets)}")

    if not targets:
        print("\nNothing to process — all entries already have video_summary filled.")
        print("\n=== Final Report ===")
        print(f"Processed: 0")
        print(f"Updated:   0")
        print(f"Failed:    0")
        print(f"Missing video files: 0")
        print(f"HolmesVAU-2B checkpoint downloaded: {checkpoint_downloaded}")
        return

    # ── Build video file map ───────────────────────────────────────────────
    video_map = build_video_map(VIDEO_SEARCH_DIRS)
    print(f"Indexed video files: {len(video_map)}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading model from {MLLM_PATH} ...")
    model, tokenizer, generation_config, sampler = load_model(MLLM_PATH, SAMPLER_PATH, DEVICE)
    print("Model loaded.")

    # Ensure deterministic
    generation_config['do_sample'] = False

    # ── Batch inference ────────────────────────────────────────────────────
    processed = []
    updated   = []
    failed    = []
    missing   = []

    for vid_id in targets:
        print(f"\n[{len(processed)+1}/{len(targets)}] {vid_id}")
        if vid_id not in video_map:
            print(f"  ✗ Video file not found — skipping")
            missing.append(vid_id)
            continue

        video_path = video_map[vid_id]
        print(f"  Video: {video_path}")
        try:
            is_normal = (len(data[vid_id].get('label', [])) == 0) or vid_id.startswith('Normal')
            if is_normal:
                current_prompt = random.choice(NORMAL_PROMPT_LIST)
            else:
                current_prompt = random.choice(PROMPT_LIST)
            print(f"  Prompt: {current_prompt}")
            pred, history, frame_indices, anomaly_score = generate(
                video_path, current_prompt,
                model, tokenizer, generation_config, sampler,
                select_frames=12, use_ATS=True
            )
            # Clean up output
            summary = pred.strip()
            data[vid_id]['video_summary'] = summary
            updated.append(vid_id)
            processed.append(vid_id)
            print(f"  ✓ Summary: {summary[:120]}...")

            # Save incrementally
            with open(JSON_PATH, 'w') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"  ✗ Inference failed: {e}")
            traceback.print_exc()
            failed.append(vid_id)
            processed.append(vid_id)

    # ── Final report ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("=== Final Report ===")
    print("=" * 60)
    print(f"Processed:              {len(processed)}")
    print(f"Updated (success):      {len(updated)}")
    print(f"Failed (inference err): {len(failed)}")
    print(f"Missing video files:    {len(missing)}")
    print(f"Checkpoint downloaded:  {checkpoint_downloaded}")
    if updated:
        print(f"\nUpdated video IDs: {updated}")
    if failed:
        print(f"\nFailed video IDs: {failed}")
    if missing:
        print(f"\nMissing video files: {missing}")

    print(f"\nJSON saved to: {JSON_PATH}")


if __name__ == '__main__':
    main()
