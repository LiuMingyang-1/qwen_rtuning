# Qwen R-Tuning

这套代码是给 `Qwen/Qwen2.5-7B-Instruct` 单独写的新训练链路，不依赖旧仓库里的 LMFlow 脚本。

你可以直接把整个 `qwen_rtuning/` 目录单独复制走。

它只做两件事：

- 用基座模型先构造 R-Tuning 风格训练数据
- 用 `Transformers + PEFT` 跑 LoRA/QLoRA SFT

## 目录

- `build_dataset.py`: 从 `R-Tuning-data` 原始数据构造新的对话式训练集
- `train.py`: 用 LoRA/QLoRA 微调 `Qwen2.5-7B-Instruct`
- `requirements.txt`: 依赖清单

## 环境

建议 Python 3.10+，CUDA 环境可用。

```bash
cd /path/to/workdir
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r qwen_rtuning/requirements.txt
```

如果你服务器已经有现成环境，只需要安装这些包即可。

## 数据准备

默认推荐把原始数据直接放在 `qwen_rtuning/R-Tuning-data/`。你现在这份代码已经按这个布局整理好了。

目录结构应当类似：

```text
qwen_rtuning/
  build_dataset.py
  train.py
  requirements.txt
  R-Tuning-data/
  pararel/training_data.json
  MMLU/MMLU_ID_train.json
  MMLU/MMLU_ID_prompt.json
  FEVER/fever_10k.json
  HotpotQA/hotpot_10k.json
  WiCE/wice_train.json
```

当前脚本支持任务：

- `pararel`
- `mmlu`
- `fever`
- `hotpotqa`
- `wice`

## 先构造训练集

### 1. `unsure` 方法

这会先让基座模型答题：

- 答对 -> 标 `I am sure.`
- 答错 -> 标 `I am unsure.`

```bash
python3 qwen_rtuning/build_dataset.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --data_root qwen_rtuning/R-Tuning-data \
  --tasks pararel mmlu fever hotpotqa wice \
  --method unsure \
  --output_path outputs/data/qwen_rtuning_unsure.jsonl \
  --load_in_4bit \
  --prompt_domain ID
```

### 2. `unknown` 方法

这会先让基座模型答题：

- 答对 -> 保留标准答案
- 答错 -> 换成一条拒答语句

```bash
python3 qwen_rtuning/build_dataset.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --data_root qwen_rtuning/R-Tuning-data \
  --tasks pararel mmlu fever hotpotqa wice \
  --method unknown \
  --output_path outputs/data/qwen_rtuning_unknown.jsonl \
  --load_in_4bit \
  --prompt_domain ID
```

### 3. `uncertain` 方法

这个版本做了适配：

- 分类任务直接用候选答案分布熵
- 开放问答任务用多次采样答案的经验熵
- 每个任务内部按熵排序，前半标 `I am sure.`，后半标 `I am unsure.`

```bash
python3 qwen_rtuning/build_dataset.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --data_root qwen_rtuning/R-Tuning-data \
  --tasks pararel mmlu fever hotpotqa wice \
  --method uncertain \
  --num_uncertainty_samples 5 \
  --output_path outputs/data/qwen_rtuning_uncertain.jsonl \
  --load_in_4bit \
  --prompt_domain ID
```

### 输出格式

输出是 JSONL，每行一个训练样本，核心字段是 `messages`，可以直接喂给 `train.py`。

同时会额外写一个统计文件：

```text
outputs/data/qwen_rtuning_unsure.jsonl.summary.json
```

里面会记录样本总数和各任务条数。

## 开始训练

### 单卡 QLoRA

```bash
python3 qwen_rtuning/train.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset_path outputs/data/qwen_rtuning_unsure.jsonl \
  --output_dir outputs/checkpoints/qwen2.5-7b-instruct-rtuning-unsure \
  --load_in_4bit \
  --bf16 \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --logging_steps 10 \
  --save_steps 200
```

### 多卡 QLoRA

```bash
torchrun --nproc_per_node=4 qwen_rtuning/train.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset_path outputs/data/qwen_rtuning_unsure.jsonl \
  --output_dir outputs/checkpoints/qwen2.5-7b-instruct-rtuning-unsure \
  --load_in_4bit \
  --bf16 \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --logging_steps 10 \
  --save_steps 200
```

## 常用参数

### `build_dataset.py`

- `--limit_per_task 100`
  先小样本试跑，确认数据构造逻辑没问题
- `--load_in_4bit`
  降低构造数据时的显存占用
- `--prompt_domain ID|OOD`
  只影响 MMLU prompt 文件选择

### `train.py`

- `--load_in_4bit`
  开启 QLoRA
- `--bf16`
  Ampere/Hopper 上优先开
- `--max_length`
  推荐从 `2048` 或 `4096` 起试
- `--target_modules`
  默认已经适配 Qwen 常见线性层

## 训练产物

`train.py` 默认保存 LoRA adapter 和 tokenizer 到 `--output_dir`。

如果你后面想合并权重，可以再单独写 merge 脚本，或者直接在推理时加载 adapter。

## 评测

仓库现在提供独立的 `eval.py`，用于加载基座模型或 `基座 + LoRA adapter`，并在测试集上做离线评测。

默认输出：

- `predictions.jsonl`：逐样本预测结果
- `metrics.json`：整体和分任务指标

### 评测训练后的 LoRA adapter

```bash
python3 qwen_rtuning/eval.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/checkpoints/qwen2.5-rtuning-unsure \
  --data_root qwen_rtuning/R-Tuning-data \
  --tasks pararel mmlu fever hotpotqa wice \
  --prompt_domain ID \
  --output_dir outputs/eval/qwen2.5-rtuning-unsure-id \
  --load_in_4bit \
  --bf16
```

### 只评测基座模型

```bash
python3 qwen_rtuning/eval.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --data_root qwen_rtuning/R-Tuning-data \
  --tasks pararel mmlu fever hotpotqa wice \
  --prompt_domain ID \
  --output_dir outputs/eval/qwen2.5-base-id \
  --load_in_4bit \
  --bf16
```

### 小样本 smoke test

```bash
python3 qwen_rtuning/eval.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/checkpoints/qwen2.5-rtuning-unsure \
  --data_root qwen_rtuning/R-Tuning-data \
  --tasks mmlu fever \
  --prompt_domain ID \
  --limit_per_task 5 \
  --output_dir outputs/eval/smoke \
  --load_in_4bit \
  --bf16
```

## 两个拒答基线（Qwen2.5-7B-Instruct 基座）

新增脚本：

- `qwen_rtuning/baseline_reject_eval.py`：在 GPU 机器上跑新基线逐样本结果
- `qwen_rtuning/icr_analysis/compare_baselines.py`：本地统一切分、阈值选择和最终对比

说明：`compare_baselines.py` 与现有 ICR 脚本口径一致，使用 `scikit-learn` 的 `StandardScaler + LogisticRegression`。

### 1) uncertainty-threshold reject（云端 GPU）

```bash
python3 qwen_rtuning/baseline_reject_eval.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --data_root qwen_rtuning/R-Tuning-data \
  --tasks pararel hotpotqa \
  --prompt_domain ID \
  --baseline uncertainty \
  --output_dir outputs/eval/baselines/uncertainty \
  --load_in_4bit \
  --bf16
```

### 2) 4-sample consistency reject（云端 GPU）

```bash
python3 qwen_rtuning/baseline_reject_eval.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --data_root qwen_rtuning/R-Tuning-data \
  --tasks pararel hotpotqa \
  --prompt_domain ID \
  --baseline consistency \
  --num_samples 4 \
  --temperature 0.7 \
  --top_p 0.95 \
  --output_dir outputs/eval/baselines/consistency \
  --load_in_4bit \
  --bf16
```

### 3) 20 samples/task smoke test（云端 GPU）

```bash
python3 qwen_rtuning/baseline_reject_eval.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --data_root qwen_rtuning/R-Tuning-data \
  --tasks pararel hotpotqa \
  --prompt_domain ID \
  --baseline consistency \
  --num_samples 4 \
  --temperature 0.7 \
  --top_p 0.95 \
  --limit_per_task 20 \
  --output_dir outputs/eval/baselines/consistency-smoke \
  --load_in_4bit \
  --bf16
```

### 4) 本地统一对比与阈值选择

`compare_baselines.py` 会：

- 在共有 ID 上生成固定 `seed=42`、`val_ratio=0.3` 的共享切分
- 在 val 上为 uncertainty / consistency 选阈值
- 在 test 上输出 `base / r-tuning_only / icr_only / icr+r-tuning(OR) / uncertainty / consistency` 指标

默认不允许 probe 训练使用 test 子集；仅 smoke/debug 时可额外加
`--allow_probe_train_on_all_common_for_debug`。

```bash
python3 qwen_rtuning/icr_analysis/compare_baselines.py \
  --base_predictions_path outputs/eval/unsure-id-base/predictions.jsonl \
  --rtuning_predictions_path outputs/eval/unsure-id-rtuning/predictions.jsonl \
  --icr_base_path qwen_rtuning/icr_analysis/outputs/outputs/icr_scores_base.jsonl \
  --icr_rtuning_path qwen_rtuning/icr_analysis/outputs/outputs/icr_scores_rtuning.jsonl \
  --uncertainty_predictions_path outputs/eval/baselines/uncertainty/predictions.jsonl \
  --consistency_predictions_path outputs/eval/baselines/consistency/predictions.jsonl \
  --tasks pararel hotpotqa \
  --output_dir outputs/eval/baselines/comparison
```

## 建议的实际执行顺序

上服务器以后先别直接全量跑，建议按这个顺序：

1. 先对单个任务小样本构造数据
2. 抽查 JSONL 里的 `messages` 是否符合预期
3. 单卡跑 50 到 200 step 看 loss 是否正常下降
4. 再开全量数据和多卡训练

例如先做一个最小 smoke test：

```bash
python3 qwen_rtuning/build_dataset.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --data_root qwen_rtuning/R-Tuning-data \
  --tasks pararel mmlu \
  --limit_per_task 20 \
  --method unsure \
  --output_path outputs/data/smoke.jsonl \
  --load_in_4bit
```

然后训练：

```bash
python3 qwen_rtuning/train.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset_path outputs/data/smoke.jsonl \
  --output_dir outputs/checkpoints/smoke \
  --load_in_4bit \
  --bf16 \
  --max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1
```

## 已知取舍

- 这套实现故意不兼容旧 LMFlow 训练入口，目的是把 Qwen 训练链路单独做干净
- `uncertain` 方法不是逐字复刻旧脚本，而是做了更适合 Qwen 的不确定性估计
- 当前只覆盖训练数据构造和 SFT 训练，没有额外重写评测脚本

如果你后面还要，我可以在这套目录上继续补一版单独的 `eval.py`。
