#!/usr/bin/env python3
"""
UCF-Crime AR text-text retrieval and score fusion using Qwen3-VL-Embedding-2B.

Inputs:
  - Query texts: /workspace/score_fuse/UCF-AR/ucf_crime_test.json
  - Video descriptions: /workspace/outputs/holmesvau_video_descriptions_290.json
  - Visual retrieval top-k: /workspace/score_fuse/output/ucfcrime_ar_retrieval_run/topk_results.json

Outputs:
  - text_topk.json
  - alpha_*/topk.json
  - metrics.json

Notes:
  - The visual branch file only contains top-k visual scores, not a full 290x290
    similarity matrix. This script therefore treats missing visual scores as 0.0.
  - Fusion follows: fused_score = alpha * visual_score + (1 - alpha) * text_score
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


QWEN_REPO_DIR = "/workspace/score_fuse/Qwen3-VL-Embedding"
if QWEN_REPO_DIR not in sys.path:
    sys.path.insert(0, QWEN_REPO_DIR)

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder


QUERY_INSTRUCTION = "Represent the query text for retrieving the most relevant video description."
DOC_INSTRUCTION = "Represent the video description for retrieval against natural-language crime queries."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL-Embedding text retrieval and fuse with visual retrieval scores."
    )
    parser.add_argument(
        "--query_json",
        type=str,
        default="/workspace/score_fuse/UCF-AR/ucf_crime_test.json",
        help="Path to UCF-Crime AR test annotations.",
    )
    parser.add_argument(
        "--description_json",
        type=str,
        default="/workspace/outputs/holmesvau_video_descriptions_290.json",
        help="Path to Holmes-VAU video descriptions JSON.",
    )
    parser.add_argument(
        "--visual_topk_json",
        type=str,
        default="/workspace/score_fuse/output/ucfcrime_ar_retrieval_run/topk_results.json",
        help="Path to visual retrieval top-k JSON.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="Qwen3-VL-Embedding model name or local path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/score_fuse/output/ucfcrime_ar_qwen3vl_text_fusion",
        help="Directory for top-k and metrics outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k results to save per query.",
    )
    parser.add_argument(
        "--alpha_step",
        type=float,
        default=0.1,
        help="Step size for fusion alpha sweep from 0 to 1.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype passed to the embedding model.",
    )
    return parser.parse_args()


def get_torch_dtype(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def basename_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def build_description_map(description_json):
    data = load_json(description_json)
    desc_map = {}
    for item in data:
        video_name = item["video_name"]
        desc_map[video_name] = item["video_description"].strip()
    return desc_map


def build_gallery_and_queries(query_json, description_map):
    items = load_json(query_json)
    gallery = []
    queries = []

    for idx, item in enumerate(items):
        video_relpath = item["Video Name"]
        query_text = item["English Text"].strip()
        basename = basename_without_ext(video_relpath)
        if basename not in description_map:
            raise KeyError(f"Missing video description for {basename} from {video_relpath}")

        gallery.append(
            {
                "video_index": idx,
                "video_name": video_relpath,
                "video_basename": basename,
                "video_summary": description_map[basename],
            }
        )
        queries.append(
            {
                "query_index": idx,
                "query_text": query_text,
                "ground_truth_video": video_relpath,
                "ground_truth_summary": description_map[basename],
            }
        )

    return gallery, queries


def embed_texts(embedder, texts, instruction, batch_size):
    all_embeddings = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        model_inputs = [{"text": text, "instruction": instruction} for text in batch]
        embeddings = embedder.process(model_inputs)
        all_embeddings.append(embeddings.detach().cpu().to(torch.float32))
        print(f"Embedded {min(start + batch_size, total)}/{total}", end="\r")
    print()
    return torch.cat(all_embeddings, dim=0).numpy()


def compute_metrics(score_matrix):
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


def build_topk_entries(score_matrix, queries, gallery, topk):
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
                    "video_summary": doc["video_summary"],
                    "score": float(scores[video_idx]),
                    "is_ground_truth": bool(video_idx == query_idx),
                }
            )
        results.append(
            {
                "query_index": query["query_index"],
                "query_text": query["query_text"],
                "ground_truth_video": query["ground_truth_video"],
                "ground_truth_summary": query["ground_truth_summary"],
                "topk": topk_entries,
            }
        )
    return results


def load_visual_score_matrix(visual_topk_json, num_queries, num_videos):
    visual_data = load_json(visual_topk_json)
    matrix = np.zeros((num_queries, num_videos), dtype=np.float32)
    for item in visual_data:
        qidx = int(item["query_index"])
        for cand in item["topk"]:
            vidx = int(cand["video_index"])
            matrix[qidx, vidx] = float(cand["score"])
    return matrix


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    print("=" * 80)
    print("UCF-Crime AR Qwen3-VL Text Fusion")
    print("=" * 80)
    print(f"Query JSON        : {args.query_json}")
    print(f"Description JSON  : {args.description_json}")
    print(f"Visual top-k JSON : {args.visual_topk_json}")
    print(f"Model             : {args.model_name_or_path}")
    print(f"Output dir        : {args.output_dir}")
    print(f"Device            : {args.device}")

    description_map = build_description_map(args.description_json)
    gallery, queries = build_gallery_and_queries(args.query_json, description_map)

    print(f"Queries  : {len(queries)}")
    print(f"Gallery  : {len(gallery)}")

    device = torch.device(args.device)
    embedder = Qwen3VLEmbedder(
        model_name_or_path=args.model_name_or_path,
        torch_dtype=get_torch_dtype(args.torch_dtype),
    )
    embedder.model = embedder.model.to(device)

    query_texts = [item["query_text"] for item in queries]
    doc_texts = [item["video_summary"] for item in gallery]

    print("Encoding query texts...")
    query_embeddings = embed_texts(embedder, query_texts, QUERY_INSTRUCTION, args.batch_size)
    print("Encoding document texts...")
    doc_embeddings = embed_texts(embedder, doc_texts, DOC_INSTRUCTION, args.batch_size)

    text_scores = np.matmul(query_embeddings, doc_embeddings.T).astype(np.float32)
    visual_scores = load_visual_score_matrix(
        args.visual_topk_json,
        num_queries=len(queries),
        num_videos=len(gallery),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_topk = build_topk_entries(text_scores, queries, gallery, args.topk)
    save_json(output_dir / "text_topk.json", text_topk)

    alpha_values = []
    alpha = 0.0
    while alpha <= 1.0 + 1e-8:
        alpha_values.append(round(alpha, 4))
        alpha += args.alpha_step

    fusion_metrics = {}
    for alpha in alpha_values:
        fused_scores = alpha * visual_scores + (1.0 - alpha) * text_scores
        alpha_name = f"alpha_{alpha:.4f}"
        topk_entries = build_topk_entries(fused_scores, queries, gallery, args.topk)
        save_json(output_dir / alpha_name / "topk.json", topk_entries)
        fusion_metrics[f"{alpha:.4f}"] = compute_metrics(fused_scores)
        print(f"Saved fusion results for alpha={alpha:.4f}")

    metrics = {
        "text_only": compute_metrics(text_scores),
        "visual_only_from_top10": compute_metrics(visual_scores),
        "fusion": fusion_metrics,
        "best_alpha_by_recall@1": max(
            fusion_metrics.items(),
            key=lambda item: item[1]["recall@1"],
        )[0],
        "notes": {
            "visual_score_source": args.visual_topk_json,
            "fusion_rule": "alpha * visual_score + (1 - alpha) * text_score",
            "visual_score_assumption": (
                "Only top-k visual scores were available. Missing visual scores were filled with 0.0."
            ),
            "query_instruction": QUERY_INSTRUCTION,
            "document_instruction": DOC_INSTRUCTION,
            "model_name_or_path": args.model_name_or_path,
        },
    }
    save_json(output_dir / "metrics.json", metrics)

    print("\nDone.")
    print(f"text_topk.json : {output_dir / 'text_topk.json'}")
    print(f"metrics.json   : {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
