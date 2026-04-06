#!/venv/main/bin/python3
"""
Single video HolmesVAU inference.
Run this script and pass the path to the video file you want to process.
"""

import os
import sys
import torch
import argparse

# Add HolmesVAU to path
HOLMESVAU_DIR = '/workspace/Capstone/HolmesVAU'
sys.path.insert(0, HOLMESVAU_DIR)
os.chdir(HOLMESVAU_DIR)  # needed for relative imports in holmesvau package

from holmesvau.holmesvau_utils import load_model, generate

# ── Paths ──────────────────────────────────────────────────────────────────
MLLM_PATH    = '/workspace/Capstone/HolmesVAU/ckpts/HolmesVAU-2B'
SAMPLER_PATH = '/workspace/Capstone/HolmesVAU/holmesvau/ATS/anomaly_scorer.pth'

PROMPT = (
    "Provide a summary of the anomaly events in the video."
)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser(description="Run HolmesVAU on a single video.")
    parser.add_argument("video_path", type=str, help="Absolute path to the video file to process.")
    args = parser.parse_args()

    video_path = args.video_path

    if not os.path.isfile(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        sys.exit(1)

    print("=" * 60)
    print("HolmesVAU Single Video Inference")
    print("=" * 60)

    print(f"Loading model from {MLLM_PATH} ...")
    model, tokenizer, generation_config, sampler = load_model(MLLM_PATH, SAMPLER_PATH, DEVICE)
    print("Model loaded.")

    # Ensure deterministic
    generation_config['do_sample'] = False

    print(f"\nProcessing Video: {video_path}")
    print(f"Prompt: {PROMPT}\n")
    
    try:
        pred, history, frame_indices, anomaly_score = generate(
            video_path, PROMPT,
            model, tokenizer, generation_config, sampler,
            select_frames=12, use_ATS=True
        )
        import matplotlib
        matplotlib.use('Agg') # Safe for headless server
        import matplotlib.pyplot as plt
        from decord import VideoReader, cpu
        from holmesvau.holmesvau_utils import show_smapled_video
        import numpy as np

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", os.path.splitext(os.path.basename(video_path))[0])
        os.makedirs(out_dir, exist_ok=True)

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        
        # Manually perform what show_smapled_video does so we can save it properly
        all_frame = []
        for i in frame_indices:
            frame = vr[i].asnumpy()
            all_frame.append(frame)
        h, w, c = all_frame[0].shape
        frame_show = np.zeros((h, w*len(all_frame), c), dtype=np.uint8)
        for i in range(len(all_frame)):
            frame_show[:, i*w:(i+1)*w, :] = all_frame[i]
            frame_show[:, i*w:i*w+5, :] = 255

        plt.figure(figsize=(20, 10))
        plt.imshow(frame_show)
        plt.axis('off')
        frames_out_path = os.path.join(out_dir, "sampled_frames.png")
        plt.savefig(frames_out_path, bbox_inches='tight')
        plt.close()

        if anomaly_score is not None:
            plt.figure(figsize=(8, 2))
            plt.plot(anomaly_score)
            for idx in frame_indices:
                plt.vlines(idx/16, 0, 1, colors='r')
            plt.ylim(0, 1)
            plt.xlabel('snippet')
            plt.ylabel('anomaly score')
            score_out_path = os.path.join(out_dir, "anomaly_score.png")
            plt.savefig(score_out_path, bbox_inches='tight')
            plt.close()

        print("=" * 60)
        print("GENERATED SUMMARY/DESCRIPTION:")
        print("=" * 60)
        print(f"Selected Frames (ATS): {frame_indices}")
        print(f"Saved frames plot to: {frames_out_path}")
        if anomaly_score is not None:
            print(f"Saved anomaly score plot to: {score_out_path}")
        print("-" * 60)
        print(pred.strip())
        print("=" * 60)
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == '__main__':
    main()
