# Inference and evaluation

## Install and download

Install a CUDA-compatible PyTorch build, then from the DeepGrounder repository root:

```bash
pip install -r requirements-inference.txt
huggingface-cli download Apollo-ULTA/DeepGrounder --local-dir checkpoints/DeepGrounder
```

The checkpoint is based on Qwen2.5-VL-3B and has two Safetensors shards. Keep all shard, index, tokenizer, chat-template, and preprocessor files together. It is a full model checkpoint, so you do not need to load a separate base model or adapter.

## Prepare DIOR-RSVG

Download the [official dataset](https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_?usp=sharing) and retain the official split files. From the repository root:

```bash
python tools/prepare_dior_rsvg.py \
  --data_root data/DIOR_RSVG \
  --output_dir data/processed/DIOR-RSVG
```

The converter produces `test_eval.jsonl`, with one example per line:

```json
{"file_name":"00001.jpg","sent":"the ship near the harbor","bbox":[100,120,240,300]}
```

This is a schema example, not a benchmark annotation. `bbox` is `[xmin, ymin, xmax, ymax]` in the **original image's pixel coordinates**. The test set is for evaluation only.

To evaluate your own labeled images, create JSONL records with these same keys and point `--image_base_path` at their image directory. A one-line JSONL file can be used for a single labeled example.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval/infer_allbox.py \
  --reasoning_model_path checkpoints/DeepGrounder \
  --json_path data/processed/DIOR-RSVG/test_eval.jsonl \
  --image_base_path data/DIOR_RSVG/JPEGImages \
  --output_dir outputs/DIOR-RSVG \
  --resize_size 840 \
  --max_new_tokens 1024
```

The script uses greedy decoding and requests structured spatial reasoning followed by an answer such as:

```text
<think>...</think>
<answer>{"bbox": [x1, y1, x2, y2]}</answer>
```

The example indicates the output format; generated explanations should be checked against the image.

### Coordinates and resolution

The existing evaluator resizes each input to a square of `resize_size` pixels (840 by default). Predicted coordinates are interpreted on that resized image, then mapped back with **separate horizontal and vertical scale factors** before evaluation. Do not convert pixel coordinates into the 0–1000 normalized format.

Resolution and prompt selection affect the results. The command above exposes the repository evaluator's preprocessing convention; it is not a guarantee of reproducing every paper table. Preserve a fixed prompt, image resolution, model revision, and decoding setup when comparing checkpoints.

## Outputs and metrics

- `metrics.json`: Pr@0.5 / 0.6 / 0.7 / 0.8 / 0.9, mean IoU, and overall IoU.
- `error_cases.txt`: examples for which no valid box was parsed.
- Terminal: model output parsing and final metric summary.

Saved metrics are fractions in [0, 1]; multiply by 100 to compare with the paper's percentage tables. Pr@k is the fraction of examples with IoU ≥ k. Mean IoU averages per-example overlap; overall IoU divides total intersection by total union. Malformed/missing boxes contribute zero IoU and remain in the denominator.

The legacy `--result_images_dir` option is retained for compatibility; this evaluator does not currently save annotated images. Paper visualizations are provided in the main README.

## Troubleshooting

- **Model architecture not found:** use Transformers 4.49.0 as specified in the inference requirements.
- **CUDA out of memory:** close other GPU jobs or reduce `--max_new_tokens` or `--resize_size`; record changed settings because they can affect metrics.
- **Missing images:** check `file_name` against the image root.
- **Wrong input keys:** use `test_eval.jsonl`, not `test_grpo.jsonl`.
- **Unexpected scores:** confirm checkpoint identity, split files, preprocessing, and the denominator. Paper values are reported references; the uploaded checkpoint has not been re-benchmarked as part of this documentation update.
