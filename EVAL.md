ConsistCompose-BAGEL Model Benchmark Evaluation

# 1. Model Benchmark Overview

This document describes the benchmark evaluation pipeline for **ConsistCompose-BAGEL** model on two standard benchmarks: **COCO-Position** and **MSBench**.

- **COCO-Position**: Text-to-image generation benchmark with normalized bounding box constraints, evaluating the model's ability to generate objects at specified spatial positions.

- **MSBench**: Multi-subject driven image generation benchmark, evaluating the model's ability to compose multiple reference subjects with precise layout control.



# 2 Benchmark Evaluation

## 2.1 COCO-Position Benchmark

Evaluate layout-controlled text-to-image generation with normalized bounding box annotations from COCO dataset.

**Run Command**:

```bash

python eval/eval_coco_position.py \

--model_path sensenova/ConsistCompose-BAGEL-7B-MoT \

--mode layout_t2i \

--coco_position_benchmark_json_path ./eval/data/coco_position.jsonl \

--output_dir ./eval/result/coco_position/

```

**Output Structure**:

```bash

eval/result/coco_position/

├── bagel_{coco_id}.jpg        # Raw generated images

└── vis/                       # Bounding box visualization results

    └── bagel_{coco_id}.jpg

```

### MIG-Bench Based Evaluation

This section details the configuration and execution process for COCO POSITION evaluation based on MIG-Bench (CVPR2024 MIGC benchmark).

**Table of Contents**: Overview, Env & Repo Preparation, Execution Commands, Parameter Explanation

**Overview**: COCO POSITION evaluation leverages MIG-Bench's core capabilities to evaluate multi-instance position control for text-to-image synthesis.

**Env & Repo Preparation**:

First, navigate to the `eval` directory (create it if not exists) and clone the MIG-Bench repository
The entire environment configuration and pretrained model weights download strictly follow **the official Installation section of MIG-Bench's README**.
```bash

# Create and enter the eval directory (create if not exists)

cd eval

# Clone MIG-Bench repository

git clone https://github.com/LeyRio/MIG_Bench.git

cd MIG_Bench

```

Place the `coco_position_benchmark.py` evaluation script to the root directory of MIG-Bench (adjust the path as needed):

```bash

# Move the script to MIG-Bench root directory (replace with your actual script path)

mv ../coco_position_benchmark.py coco_position_benchmark.py

```

**Execution Commands**:

Execute the evaluation script under the root directory of MIG-Bench (ensure you are in `eval/MIG_Bench/`):

```bash

# Define paths (relative to MIG-Bench root)

COCO_POSITION_JSON="{PROJECT_ROOT}/eval/data/coco_position.jsonl"

COCO_OUT_DIR="{PROJECT_ROOT}/eval/result/coco_position/"

mkdir -p ${COCO_OUT_DIR}

# Run evaluation

python coco_position_benchmark.py \

    --coco_path ${COCO_POSITION_JSON} \

    --gen_dir ${COCO_OUT_DIR} \

    --out_json ${COCO_OUT_DIR}/coco_position_reuslt.json

```

## 2.2 MSBench Benchmark
#### Description
Multi-subject composition benchmark, testing the model's capability to generate images with multiple reference objects and specified layout constraints.

### MSBench 
#### Run Command
```bash
python eval/eval_msbench.py \
--model_path sensenova/ConsistCompose-BAGEL-7B-MoT \
--mode layout_subject_driven \
--out_img_dir eval/result/msbench/ \
--msbench_jsonl_path ./eval/data/msbench.jsonl \
--msbench_input_dir {PROJECT_ROOT}/eval/data/
```
> Replace `{PROJECT_ROOT}` with your actual project root path.

#### Output Structure
```
eval/result/msbench/
├── 00000_comb_type.jpg        # Raw generated images
└── vis/                       # Combined visualization (ref images + generated image + bboxes)
    └── 00000_comb_type_combined.jpg
```

### MSBench Random
#### Run Command
```bash
python eval/eval_msbench.py \
--model_path sensenova/ConsistCompose-BAGEL-7B-MoT \
--mode layout_subject_driven \
--out_img_dir eval/result/msbench_random/ \
--msbench_jsonl_path ./eval/data/msbench_random.jsonl \
--msbench_input_dir {PROJECT_ROOT}/eval/data/
```
> Replace `{PROJECT_ROOT}` with your actual project root path.

#### Output Structure
```
eval/result/msbench_random/
├── 00000_comb_type.jpg        # Raw generated images
└── vis/                       # Combined visualization (ref images + generated image + bboxes)
    └── 00000_comb_type_combined.jpg
```

### MSBench Metric Evaluation

After generating images for MSBench, you can compute benchmark metrics with `msbench_benchmark.py`.

#### Description

This script evaluates generated MSBench images against the corresponding input JSONL file aligned with the generation order, and exports the metric results as a CSV file.

#### Run Command

```bash
python eval/msbench_benchmark.py \
    --jsonl ./eval/data/msbench.jsonl \
    --gen_dir ./eval/result/msbench/ \
    --out_csv ./eval/result/msbench/clip_scores.csv
```

For the random-version benchmark:

```bash
python eval/msbench_benchmark.py \
    --jsonl ./eval/data/msbench_random.jsonl \
    --gen_dir ./eval/result/msbench_random/ \
    --out_csv ./eval/result/msbench_random/clip_scores.csv
```
### Output

The evaluation script saves a CSV file containing the metric results, for example:

```bash
eval/result/msbench/
├── 00000_comb_type.jpg
├── ...
├── vis/
│   └── 00000_comb_type_combined.jpg
└── clip_scores.csv
```

## Visualization Features
- **COCO-Position**: Automatic bounding box drawing on generated images
- **MSBench**: 
  1. Reference subject images with tags
  2. Generated target image with labeled bounding boxes
  3. Horizontal composition of all images for comparison

## Notes
1. Use `--mode layout_t2i` for COCO-Position (pure layout text-to-image)
2. Use `--mode layout_subject_driven` for MSBench (subject-driven layout generation)
3. All outputs are saved in separate directories for clear result management
4. Random seed is fixed for reproducible evaluation results