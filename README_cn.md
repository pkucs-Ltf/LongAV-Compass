# LongAV-Compass

LongAV-Compass 是一个面向长视频音视频生成的评测框架，支持 T2AV、I2AV 和 V2AV。它评测事件级 QA、视觉质量、长视频结构、转场稳定性、整体观感、文本-视频对齐、图像对齐，以及可选的音频诊断指标。

这个仓库只放代码、配置模板和小规模示例。完整数据集、固定 QA 文件、模型生成视频、评测结果表、缓存和真实密钥不应该提交到 Git 仓库。

[数据集](https://huggingface.co/datasets/TengfeiLiuCoder/LongAV-Compass) | [环境文件](environment.yaml) | [分数定义](docs/score_definition.md) | [数据格式](docs/dataset_format.md) | [故障排查](docs/troubleshooting.md)

**Hugging Face 数据集地址：** https://huggingface.co/datasets/TengfeiLiuCoder/LongAV-Compass

## 快速开始

```bash
git clone https://github.com/pkucs-Ltf/LongAV-Compass.git
cd LongAV-Compass

conda env create -f environment.yaml
conda activate longav_compass

bash setup_longav.sh

# 在 configs/api_keys.yaml 中填写本地可用的 API key 或 Vertex 配置。
```

`environment.yaml` 会安装 Python、ffmpeg、PyTorch、judge 依赖和 OpenAI CLIP。`setup_longav.sh` 会安装本仓库，自动下载 ViT-B/32 CLIP 权重，并创建 `configs/api_keys.yaml`。CLIP/EventCLIP/ImageCLIP 指标建议使用本地 NVIDIA GPU；只做分数聚合或 API judge 评测不需要 GPU。

下载数据集：

```bash
huggingface-cli download TengfeiLiuCoder/LongAV-Compass \
  --repo-type dataset \
  --local-dir data/LongAV-Compass
```

把你自己的模型生成视频放到 `outputs/`。模型目录名里要包含任务名：

```text
outputs/
  MyModel_v2av_run/
    advertising/
      V2AV_001/
        full_video.mp4
```

构建 prepared samples 并运行评测：

```bash
PYTHONPATH=src python -m longav_eval prepare-event-testset \
  --source-root data/LongAV-Compass \
  --output-root outputs \
  --test-root samples/prepared \
  --sample V2AV_001

LONGAV_QA_ROOT=data/LongAV-Compass \
bash run_eval_batch.sh samples/prepared runs/example_eval '*__*'
```

可以使用 `--sample T2AV_001`、`--sample I2AV_001` 或 `--sample V2AV_001`；重复写 `--sample` 可以一次构建多个样本。

从宽表计算论文中的百分制 balanced score：

```bash
bash run_compute_scores.sh /path/to/best_sample_model_metrics_wide.csv runs/paper
```

## 样本格式

LongAV-Compass 评测的是已经整理好的 sample directory：

```text
sample_dir/
  sample_index.json
  canonical_events.json
  event_checklists.json          可选
  <model_name>/
    events_manifest.json
    full_video.mp4
    events/
      event_001.mp4
      event_002.mp4
    boundaries/
      boundary_001.mp4
```

具体字段见 [docs/dataset_format.md](docs/dataset_format.md)，最小模板见 [examples/README.md](examples/README.md)。

使用 LongAVBench 发布的固定 QA 时，通过 `LONGAV_QA_ROOT` 指定数据集根目录：

```bash
LONGAV_QA_ROOT=data/LongAV-Compass \
bash run_eval_batch.sh samples/prepared runs/with_fixed_qa '*__*'
```

如果不指定 `LONGAV_QA_ROOT`，评测器会优先使用样本目录内的 `event_checklists.json`；如果也没有，则通过配置的 judge provider 生成 fallback checklist。

## 常用命令

查看指标和 profile：

```bash
PYTHONPATH=src python -m longav_eval list-metrics
PYTHONPATH=src python -m longav_eval list-profiles
```

只评某些模型：

```bash
LONGAV_MODEL_NAMES="Kling Seedance2 Veo" \
LONGAV_MAX_WORKERS=8 \
bash run_eval_batch.sh samples/prepared runs/selected_models 'v2av__*'
```

基于已有结果补评缺失指标：

```bash
LONGAV_REUSE_ROOT=runs/previous_eval \
LONGAV_ONLY_MISSING=1 \
bash run_eval_batch.sh samples/prepared runs/resume_eval '*__*'
```

打开音频诊断：

```bash
LONGAV_SKIP_AUDIO=0 bash run_eval_batch.sh samples/prepared runs/full_eval '*__*'
```

## 输出位置

批量评测的输出目录包含：

```text
output_root/
  batch_status.json
  all_model_scores.csv
  <sample_id>/
    event_eval_summary.json
    model_scores.csv
    event_fulfillment_scores.csv
    event_realization_scores.csv
    longform_scores.csv
    transition_scores.csv
    holistic_scores.csv
    text_video_alignment.csv
    <model_name>/
      model_summary.json
```

分数聚合脚本输出：

```text
runs/paper/
  balanced_sample_model_scores.csv
  scenario_balanced_scores.csv
```

## 分数口径

论文中的主表和场景、难度、事件数分析使用 `balanced_score`：六个共享视频指标先缩放到 0-100，再等权平均。具体定义见 [docs/score_definition.md](docs/score_definition.md)。

人工一致性实验不使用总分，而是比较三个维度的 pairwise win rate。

## 仓库结构

```text
src/longav_eval/        Python 包和 CLI
scripts/                setup、批量评测、分数聚合包装脚本
configs/                profile 和 API key 模板
examples/               最小样本模板
data/manifests/         旧 pipeline 的 manifest 示例
docs/                   数据格式、分数定义、复现实验、故障排查
```

## 引用

论文发布后会补充 citation。
