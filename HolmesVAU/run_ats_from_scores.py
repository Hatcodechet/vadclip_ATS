import argparse
from pathlib import Path

import numpy as np

from holmesvau.ATS.Temporal_Sampler import Temporal_Sampler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Holmes-VAU ATS directly from a precomputed temporal score sequence."
    )
    parser.add_argument(
        "--score_path",
        type=str,
        required=True,
        help="Path to a .npy file containing a 1D temporal score sequence, e.g. fused.npy.",
    )
    parser.add_argument(
        "--select_frames",
        type=int,
        default=16,
        help="Number of temporal indices to sample.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Density smoothing term added before cumulative-score interpolation.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional output path for sampled indices as a .npy file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    score_path = Path(args.score_path)
    scores = np.load(score_path)

    sampler = Temporal_Sampler(ckpt_path=None, device="cpu", tau=args.tau)
    prepared_scores, sampled_idxs = sampler.sample_from_scores(
        scores,
        select_frames=args.select_frames,
        return_scores=True,
    )

    print(f"Loaded scores from: {score_path}")
    print(f"Score length: {prepared_scores.shape[0]}")
    print(f"Selected indices ({len(sampled_idxs)}): {sampled_idxs}")

    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, np.asarray(sampled_idxs, dtype=np.int64))
        print(f"Saved sampled indices to: {save_path}")


if __name__ == "__main__":
    main()
