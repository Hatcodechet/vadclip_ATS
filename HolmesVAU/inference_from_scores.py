import argparse
import json
from pathlib import Path

import numpy as np
import torch

from holmesvau.holmesvau_utils import load_model, generate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Holmes-VAU description inference from a video and precomputed temporal scores."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the source video.",
    )
    parser.add_argument(
        "--score_path",
        type=str,
        required=True,
        help="Path to a .npy score file such as fused.npy.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/score_fuse/HolmesVAU/ckpts/HolmesVAU-2B",
        help="Path to the HolmesVAU-2B model directory.",
    )
    parser.add_argument(
        "--sampler_path",
        type=str,
        default=None,
        help="Optional URDMU checkpoint. Not needed for score-driven ATS.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Could you specify the anomaly events present in the video?",
        help="Prompt sent to Holmes-VAU.",
    )
    parser.add_argument(
        "--select_frames",
        type=int,
        default=12,
        help="Number of frames to select for Holmes-VAU.",
    )
    parser.add_argument(
        "--dense_sample_freq",
        type=int,
        default=16,
        help="Dense frame step used to map ATS indices back to video frames.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Density smoothing term for ATS.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for Holmes-VAU inference. Use auto, cpu, or cuda:0.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save the final result as JSON.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)

    if not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    capability = torch.cuda.get_device_capability(0)
    arch = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = set(torch.cuda.get_arch_list())
    if compiled_arches and arch not in compiled_arches:
        print(
            f"CUDA device architecture {arch} is not supported by the current PyTorch build. "
            "Falling back to CPU."
        )
        return torch.device("cpu")

    return torch.device("cuda:0")


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    score_path = Path(args.score_path)
    scores = np.load(score_path)

    device = resolve_device(args.device)
    model, tokenizer, generation_config, sampler = load_model(
        args.model_path,
        sampler_path=args.sampler_path,
        device=device,
        sampler_tau=args.tau,
    )
    generation_config["max_new_tokens"] = args.max_new_tokens

    pred, history, frame_indices, anomaly_score = generate(
        video_path=str(video_path),
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        sampler=sampler,
        dense_sample_freq=args.dense_sample_freq,
        select_frames=args.select_frames,
        use_ATS=True,
        external_scores=scores,
    )

    result = {
        "video_path": str(video_path),
        "score_path": str(score_path),
        "select_frames": args.select_frames,
        "dense_sample_freq": args.dense_sample_freq,
        "tau": args.tau,
        "sampled_frame_indices": [int(idx) for idx in frame_indices],
        "description": pred,
    }

    print("Sampled frame indices:", result["sampled_frame_indices"])
    print("Description:", result["description"])

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("Saved result to:", output_path)


if __name__ == "__main__":
    main()
