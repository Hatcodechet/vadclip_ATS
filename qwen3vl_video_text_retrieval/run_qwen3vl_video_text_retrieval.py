#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


QWEN_REPO_DIR = "/workspace/qwen3_vl_embedding_repo"
if QWEN_REPO_DIR not in sys.path:
    sys.path.insert(0, QWEN_REPO_DIR)

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from src.models.qwen3_vl_reranker import Qwen3VLReranker


QUERY_INSTRUCTION = "Represent the query text for retrieving the most relevant surveillance video."
VIDEO_INSTRUCTION = "Represent the surveillance video for retrieval against natural-language crime descriptions."
RERANK_INSTRUCTION = "Given a search query, retrieve the most relevant surveillance video."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL-Embedding + Qwen3-VL-Reranker video-text retrieval on UCF-Crime AR test data."
    )
    parser.add_argument(
        "--query-json",
        type=str,
        default="/workspace/vadclip_ATS/UCF-AR/ucf_crime_test.json",
        help="Path to the UCF-Crime AR test JSON.",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default="/workspace/test",
        help="Root directory containing the 290 test videos.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="Embedding model name or local path.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="Qwen/Qwen3-VL-Reranker-2B",
        help="Reranker model name or local path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/vadclip_ATS/qwen3vl_video_text_retrieval/outputs",
        help="Directory to store retrieval outputs.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=2,
        help="Batch size for Qwen3-VL video/text embedding.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k to keep for retrieval and reranking.",
    )
    parser.add_argument(
        "--rerank-topk",
        type=int,
        default=20,
        help="How many embedding-stage candidates to pass into the reranker.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype used when loading Qwen models.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation passed to the models.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Video fps sampling parameter used by Qwen3-VL processing.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=64,
        help="Maximum number of video frames used by the Qwen models.",
    )
    return parser.parse_args()


def get_torch_dtype(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_video_path(video_root: Path, rel_path: str) -> Path:
    candidate = video_root / rel_path
    if candidate.exists():
        return candidate

    fallback = video_root / Path(rel_path).name
    if fallback.exists():
        return fallback

    basename = Path(rel_path).name
    matches = [
        path for path in video_root.rglob(basename)
        if path.is_file() and not path.name.startswith("._") and "__MACOSX" not in path.parts
    ]
    if matches:
        return sorted(matches)[0]

    raise FileNotFoundError(f"Cannot resolve video path for {rel_path}")


def build_dataset(query_json: str, video_root: str):
    items = load_json(query_json)
    video_root_path = Path(video_root)

    queries = []
    gallery = []
    for idx, item in enumerate(items):
        video_relpath = item["Video Name"]
        query_text = item["English Text"].strip()
        video_path = resolve_video_path(video_root_path, video_relpath)
        video_name = Path(video_relpath).stem

        queries.append(
            {
                "query_index": idx,
                "query_text": query_text,
                "ground_truth_video": video_relpath,
                "ground_truth_video_name": video_name,
            }
        )
        gallery.append(
            {
                "video_index": idx,
                "video_name": video_name,
                "video_relpath": video_relpath,
                "video_path": str(video_path),
            }
        )

    return queries, gallery


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def embed_queries(embedder: Qwen3VLEmbedder, queries: list[dict], batch_size: int) -> np.ndarray:
    all_embeddings = []
    total = len(queries)
    for start in range(0, total, batch_size):
        batch = queries[start:start + batch_size]
        model_inputs = [
            {
                "text": item["query_text"],
                "instruction": QUERY_INSTRUCTION,
            }
            for item in batch
        ]
        embeddings = embedder.process(model_inputs)
        all_embeddings.append(embeddings.detach().cpu().to(torch.float32).numpy())
        print(f"Embedded queries {min(start + batch_size, total)}/{total}", end="\r")
    print()
    return l2_normalize(np.concatenate(all_embeddings, axis=0))


def embed_videos(
    embedder: Qwen3VLEmbedder,
    gallery: list[dict],
    batch_size: int,
    fps: float,
    max_frames: int,
) -> np.ndarray:
    all_embeddings = []
    total = len(gallery)
    for start in range(0, total, batch_size):
        batch = gallery[start:start + batch_size]
        model_inputs = [
            {
                "video": item["video_path"],
                "instruction": VIDEO_INSTRUCTION,
                "fps": fps,
                "max_frames": max_frames,
            }
            for item in batch
        ]
        embeddings = embedder.process(model_inputs)
        all_embeddings.append(embeddings.detach().cpu().to(torch.float32).numpy())
        print(f"Embedded videos {min(start + batch_size, total)}/{total}", end="\r")
    print()
    return l2_normalize(np.concatenate(all_embeddings, axis=0))


def compute_metrics(score_matrix: np.ndarray) -> dict:
    num_queries = score_matrix.shape[0]
    gt_indices = np.arange(num_queries)
    ranks = []
    for query_idx in range(num_queries):
        order = np.argsort(-score_matrix[query_idx])
        gt_rank = int(np.where(order == gt_indices[query_idx])[0][0]) + 1
        ranks.append(gt_rank)
    ranks = np.asarray(ranks, dtype=np.int32)
    return {
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def build_topk_entries(score_matrix: np.ndarray, queries: list[dict], gallery: list[dict], topk: int) -> list[dict]:
    results = []
    for query_idx, query in enumerate(queries):
        scores = score_matrix[query_idx]
        order = np.argsort(-scores)[:topk]
        topk_entries = []
        for rank, video_idx in enumerate(order, start=1):
            doc = gallery[int(video_idx)]
            topk_entries.append(
                {
                    "rank": rank,
                    "video_index": int(video_idx),
                    "video_name": doc["video_name"],
                    "video_relpath": doc["video_relpath"],
                    "score": float(scores[video_idx]),
                    "is_ground_truth": bool(video_idx == query_idx),
                }
            )
        results.append(
            {
                "query_index": query["query_index"],
                "query_text": query["query_text"],
                "ground_truth_video": query["ground_truth_video"],
                "topk": topk_entries,
            }
        )
    return results


def rerank_candidates(
    reranker: Qwen3VLReranker,
    queries: list[dict],
    gallery: list[dict],
    embedding_scores: np.ndarray,
    rerank_topk: int,
) -> np.ndarray:
    rerank_scores = np.full_like(embedding_scores, fill_value=-1.0, dtype=np.float32)
    for query_idx, query in enumerate(queries):
        candidate_indices = np.argsort(-embedding_scores[query_idx])[:rerank_topk]
        documents = [
            {
                "video": gallery[int(video_idx)]["video_path"],
            }
            for video_idx in candidate_indices
        ]
        scores = reranker.process(
            {
                "instruction": RERANK_INSTRUCTION,
                "query": {"text": query["query_text"]},
                "documents": documents,
            }
        )
        for video_idx, score in zip(candidate_indices, scores):
            rerank_scores[query_idx, int(video_idx)] = float(score)
        print(f"Reranked {query_idx + 1}/{len(queries)}", end="\r")
    print()
    return rerank_scores


def combine_rerank_with_embedding(embedding_scores: np.ndarray, rerank_scores: np.ndarray) -> np.ndarray:
    combined = embedding_scores.copy()
    mask = rerank_scores >= 0
    combined[mask] = rerank_scores[mask]
    return combined


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Qwen3-VL Video-Text Retrieval")
    print("=" * 80)
    print(f"Query JSON          : {args.query_json}")
    print(f"Video root          : {args.video_root}")
    print(f"Embedding model     : {args.embedding_model}")
    print(f"Reranker model      : {args.reranker_model}")
    print(f"Qwen repo           : {QWEN_REPO_DIR}")
    print(f"Output dir          : {output_dir}")

    queries, gallery = build_dataset(args.query_json, args.video_root)
    print(f"Queries             : {len(queries)}")
    print(f"Gallery videos      : {len(gallery)}")

    dtype = get_torch_dtype(args.torch_dtype)
    common_model_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
    }

    embedder = Qwen3VLEmbedder(
        model_name_or_path=args.embedding_model,
        fps=args.fps,
        max_frames=args.max_frames,
        **common_model_kwargs,
    )
    query_embeddings = embed_queries(embedder, queries, args.embedding_batch_size)
    video_embeddings = embed_videos(
        embedder,
        gallery,
        args.embedding_batch_size,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    embedding_scores = query_embeddings @ video_embeddings.T

    embedding_metrics = compute_metrics(embedding_scores)
    embedding_topk = build_topk_entries(embedding_scores, queries, gallery, args.topk)
    save_json(output_dir / "embedding_metrics.json", embedding_metrics)
    save_json(output_dir / "embedding_topk.json", embedding_topk)
    np.save(output_dir / "query_embeddings.npy", query_embeddings)
    np.save(output_dir / "video_embeddings.npy", video_embeddings)
    np.save(output_dir / "embedding_scores.npy", embedding_scores)

    reranker = Qwen3VLReranker(
        model_name_or_path=args.reranker_model,
        fps=args.fps,
        max_frames=args.max_frames,
        **common_model_kwargs,
    )
    rerank_scores = rerank_candidates(
        reranker,
        queries,
        gallery,
        embedding_scores,
        rerank_topk=args.rerank_topk,
    )
    combined_scores = combine_rerank_with_embedding(embedding_scores, rerank_scores)

    rerank_metrics = compute_metrics(combined_scores)
    rerank_topk = build_topk_entries(combined_scores, queries, gallery, args.topk)
    save_json(output_dir / "rerank_metrics.json", rerank_metrics)
    save_json(output_dir / "rerank_topk.json", rerank_topk)
    np.save(output_dir / "rerank_scores.npy", rerank_scores)
    np.save(output_dir / "combined_scores.npy", combined_scores)

    summary = {
        "query_json": args.query_json,
        "video_root": args.video_root,
        "embedding_model": args.embedding_model,
        "reranker_model": args.reranker_model,
        "num_queries": len(queries),
        "num_gallery_videos": len(gallery),
        "query_instruction": QUERY_INSTRUCTION,
        "video_instruction": VIDEO_INSTRUCTION,
        "rerank_instruction": RERANK_INSTRUCTION,
        "embedding_metrics": embedding_metrics,
        "rerank_metrics": rerank_metrics,
        "topk": args.topk,
        "rerank_topk": args.rerank_topk,
    }
    save_json(output_dir / "retrieval_summary.json", summary)

    print("Embedding metrics:", json.dumps(embedding_metrics, indent=2))
    print("Rerank metrics:", json.dumps(rerank_metrics, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
