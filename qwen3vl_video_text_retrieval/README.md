# Qwen3-VL Video-Text Retrieval

Video-text retrieval pipeline for the 290 UCF-Crime AR test items in:

- `/workspace/vadclip_ATS/UCF-AR/ucf_crime_test.json`

Models:

- `Qwen/Qwen3-VL-Embedding-2B`
- `Qwen/Qwen3-VL-Reranker-2B`

Local source repo used for model classes:

- `/workspace/qwen3_vl_embedding_repo`

Run:

```bash
cd /workspace/vadclip_ATS/qwen3vl_video_text_retrieval

python run_qwen3vl_video_text_retrieval.py \
  --query-json /workspace/vadclip_ATS/UCF-AR/ucf_crime_test.json \
  --video-root /workspace/test \
  --embedding-model Qwen/Qwen3-VL-Embedding-2B \
  --reranker-model Qwen/Qwen3-VL-Reranker-2B \
  --output-dir /workspace/vadclip_ATS/qwen3vl_video_text_retrieval/outputs
```

Outputs:

- `embedding_metrics.json`
- `embedding_topk.json`
- `rerank_metrics.json`
- `rerank_topk.json`
- `retrieval_summary.json`

