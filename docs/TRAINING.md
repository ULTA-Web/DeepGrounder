# Training DeepGrounder on DIOR-RSVG

This guide covers **SFT → LoRA export → GRPO**. DIOR-RSVG is the dataset used in this walkthrough. The paper also contains separate RRSIS-D experiments and cross-domain evaluation on VRSBench; do not combine the datasets or treat these training runs as interchangeable.

## 1. Environments

Use Linux or WSL2 with NVIDIA CUDA for training. The paper reports one RTX 4090 for SFT and eight RTX 4090 GPUs for GRPO. Those are reported experimental resources, not a minimum-memory guarantee.

Use separate environments for LLaMA-Factory SFT and this repository's legacy GRPO trainer. Install a CUDA-compatible PyTorch build in each environment first.

GRPO environment:

```bash
conda create -n deepgrounder-grpo python=3.10 -y
conda activate deepgrounder-grpo
# Install torch/torchvision matching your CUDA runtime.
cd /path/to/DeepGrounder
bash setup.sh
```

The training source uses internal Transformers APIs and is pinned to Transformers **4.49.0** and TRL **0.17.0**. TRL 0.17.0 is recorded in the repository's existing experiment environment snapshot; do not automatically substitute the latest TRL. The snapshot also records PyTorch 2.5.1, torchvision 0.20.1, Accelerate 1.5.2, PEFT 0.14.0, and Datasets 3.4.0. Flash Attention 2 must be built for your installed PyTorch/CUDA versions.

`setup.sh` installs the GRPO package and additional training dependencies. It is not needed for inference-only usage; use [INFERENCE.md](INFERENCE.md) for that.

## 2. Download and convert DIOR-RSVG

Use the [official project](https://github.com/ZhanYang-nwpu/RSVG-pytorch) and its [official Google Drive download](https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_?usp=sharing). Follow the dataset authors' terms.

```text
/path/to/DIOR_RSVG/
├── Annotations/      # XML, including expressions and xyxy boxes
├── JPEGImages/
├── train.txt
├── val.txt
└── test.txt
```

From the DeepGrounder root:

```bash
python tools/prepare_dior_rsvg.py \
  --data_root /path/to/DIOR_RSVG \
  --output_dir /path/to/processed/DIOR-RSVG
```

The official split IDs are **zero-based expression/object indices**, not image IDs. The converter follows sorted XML paths and object order within each XML, matching the original dataset loader. It checks duplicate/overlapping split indices and verifies that every expression belongs to a split. Expect 26,991 train, 3,829 validation, and 7,500 test expressions for the official release.

Each split produces:

| File | Consumer |
| --- | --- |
| `train_sft.json` (and val/test equivalents) | LLaMA-Factory multimodal conversation data |
| `train_grpo.jsonl` | GRPO trainer: image, problem, solution, normal_caption |
| `test_eval.jsonl` | Evaluation: file_name, sent, bbox |
| `dior_grpo.yaml` | GRPO dataset configuration with an absolute annotation path |

Coordinates are preserved as `[xmin, ymin, xmax, ymax]` in source-image pixels. Only training annotations are used by the training configuration. The converter writes absolute image paths for SFT, so regenerate it if the data is moved to another machine.

A GRPO record looks like:

```json
{"image":"00001.jpg","problem":"Please provide the bounding box coordinate of the region this sentence describes: the ship near the harbor","solution":[100,120,240,300],"normal_caption":"the ship near the harbor"}
```

This is a schema example. Do not reuse the legacy root-level `mllm_rec_json.json` blindly: it contains paths from the earlier experiment environment. Generate your own DIOR-RSVG annotations instead.

## 3. Stage I: supervised fine-tuning

Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) in its own environment using the project's installation instructions. Use a version supporting Qwen2.5-VL and the `qwen2_vl` template, and record its commit for reproducibility.

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

In the commands below, `DEEPGROUNDER_ROOT` and `PROCESSED_ROOT` must be absolute paths:

```bash
export DEEPGROUNDER_ROOT=/path/to/DeepGrounder
export PROCESSED_ROOT=/path/to/processed/DIOR-RSVG

cp "$PROCESSED_ROOT/train_sft.json" data/dior_train_sft.json
cp "$DEEPGROUNDER_ROOT/configs/dior_sft.yaml" examples/train_lora/dior_sft.yaml
cp "$DEEPGROUNDER_ROOT/configs/dior_sft_export.yaml" examples/merge_lora/dior_sft_export.yaml
```

Merge the `dior_rsvg` entry from [configs/dataset_info.dior.json](../configs/dataset_info.dior.json) into LLaMA-Factory's `data/dataset_info.json`. Preserve the existing entries. The generated data uses the ShareGPT messages format with an `images` column.

Run SFT from the LLaMA-Factory root:

```bash
llamafactory-cli train examples/train_lora/dior_sft.yaml
```

The example config implements language-side LoRA with rank 8, alpha 32, dropout 0.05, learning rate 1e-6, cosine schedule, 10% warmup, two epochs, and frozen vision/projector modules. It uses batch size 1 and gradient accumulation 16 for a single-GPU effective batch size of 16. The config is an executable starting point based on the paper settings; adapt memory-related settings to your hardware.

### Export a merged model for Stage II

```bash
llamafactory-cli export examples/merge_lora/dior_sft_export.yaml
```

The export directory defaults to `saves/Qwen2.5-VL-3B-DeepGrounder-DIOR-SFT`. Check that it contains full model weights, config, tokenizer, and processor files. The existing GRPO model selector expects the merged model directory name to contain **Qwen2.5-VL**.

## 4. Stage II: GRPO

Activate the GRPO environment and return to DeepGrounder. All paths must resolve from the training machine:

```bash
conda activate deepgrounder-grpo
cd /path/to/DeepGrounder

export MODEL_PATH=/path/to/LLaMA-Factory/saves/Qwen2.5-VL-3B-DeepGrounder-DIOR-SFT
export IMAGE_ROOT=/path/to/DIOR_RSVG/JPEGImages
export DATA_CONFIG=/path/to/processed/DIOR-RSVG/dior_grpo.yaml
export OUTPUT_DIR=/path/to/outputs/DeepGrounder-DIOR-RSVG
export NPROC_PER_NODE=8
export DEEPGROUNDER_SEMANTIC_URL=http://127.0.0.1:5000

bash src/open-r1-multimodal/run_scripts/run_grpo_rec.sh
```

The launcher uses eight GPUs by default, eight responses per query, per-device batch size 1, gradient accumulation 2, learning rate 1e-6, KL coefficient 0.04, policy clipping 0.2, temperature 0.9, prompt/completion limits 1024/256, two epochs, and seed 42. It uses BF16, Flash Attention 2, full-parameter training, and DeepSpeed ZeRO-3 Offload. Additional trainer flags can be appended to the command.

Keep the global generation batch divisible by `num_generations` if you change the GPU count or batch size. Changing resource or sampling settings changes the experiment; record it in your run configuration.

### Semantic reward service

The full method uses **format + geometric localization + semantic consistency** rewards. The current code implements geometric reward with DIoU; the registered name `accuracy` refers to that implementation.

The released trainer contains a client in `src/open-r1-multimodal/src/open_r1/trainer/clip_func.py`. It requires an external Git-RSCLIP scoring service; that server and its exact deployment are **not included in this repository**. Set `DEEPGROUNDER_SEMANTIC_URL` to a service implementing the following contract, reachable from every training worker:

```text
POST <DEEPGROUNDER_SEMANTIC_URL>/score
Content-Type: application/json
```

```json
{"inputs":[{"image_base64":"<base64 JPEG of predicted crop resized to 256x256>","normal_caption":"the referring expression"}]}
```

Response:

```json
{"rewards":[0.73]}
```

Return one numeric reward per input, in the same order, using the semantic scorer and scaling specified in the paper. Use your trusted scoring endpoint: image crops and referring text are sent to it. A nominal URL without a working scorer does not reproduce the full hybrid reward method.

The launcher checks that the semantic URL is configured. The client preserves one reward per completion (invalid boxes receive zero) and stops on service/network errors, mismatched reward counts, or non-finite scores, so a failed scorer cannot silently disable semantic training. Verify the service separately and monitor rewards before a long run; a reachable URL alone does not establish correct scoring.

### Two-reward ablation without a scoring server

To validate the data and GRPO environment without semantic scoring:

```bash
export REWARD_FUNCS="accuracy format"
bash src/open-r1-multimodal/run_scripts/run_grpo_rec.sh
```

This is a **format/geometric reward ablation**, not the paper's full DeepGrounder method. Do not attach the full-method table values to this run.

## 5. Evaluate and record the run

Follow [INFERENCE.md](INFERENCE.md), pointing `--reasoning_model_path` to the saved full checkpoint. Record the training-data release, base-model revision, SFT framework commit, merged SFT checkpoint, reward configuration, scorer model/version, software versions, preprocessing, and final checkpoint step.

The documentation update was checked with CPU regression tests for annotation conversion and inference parsing/metrics. Full SFT, GRPO, and GPU benchmark reproduction were not run during this update.
