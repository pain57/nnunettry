# nnU-Net v2 胰腺分割（AMOS22 MRI）

基于**官方 nnU-Net v2 包**（`nnunetv2==2.8.1`，Python 3.11）的 **AMOS22 MRI 胰腺分割**
实验框架。训练 / 规划 / 预处理 / 推理全部调用官方 `nnUNetv2_*` 入口，只在官方扩展点做少量
自定义（UNETR / SwinUNETR / MedNeXt 网络），**不自建 nnU-Net 克隆**。

> 当前任务：**单器官胰腺分割**（`background=0, pancreas=1`，由 AMOS22 器官标签 `10` 映射而来）。
> 未来获得胰管 / 胆管数据集后，再切换回胰胆管分割（Exp 4 两阶段、Exp 5 clDice 已暂存待用）。

## 技术路线 → 实验映射

| # | 实验 | 做法（官方机制） | 状态 |
|---|------|------------------|------|
| 1 | **基线** PlainConvUNet + 3d_fullres | 默认 `nnUNetTrainer`（Dice+CE，官方 1000 epochs） | ✅ 主基线 |
| 2 | **骨干对比** ResEnc M/L | planner `nnUNetPlannerResEncM/L` → `nnUNetResEncUNetM/LPlans` → `nnUNetTrainer` | ✅ |
| 3 | **间距研究** | 先看 nnU-Net 自动 spacing，再决定是否固定 | ✅ 后置 |
| 4 | **两阶段** ROI → duct | 需要 duct/ROI GT | ⏸ 暂停 |
| 5 | **clDice** 连续性损失 | 需要细管道 GT | ⏸ 暂停 |
| 6 | **其他模型** UNETR / SwinUNETR / MedNeXt | 自定义 trainer（关深监督 + 2.8.1 接口） | ✅ 后置 |

## 数据集：`Dataset150_AMOSMRI_Pancreas`

| 项 | 值 |
|----|----|
| 模态 | **MRI**（不是 CT） |
| 标签 | `background=0, pancreas=1` |
| 来源 | AMOS22 MRI（器官标签 `10` → `1`） |
| 目录 | `imagesTr / labelsTr / imagesTs / labelsTs + dataset.json` |

## 安装

```bash
# Python 3.11
pip install -r requirements.txt

# MedNeXt（Exp 6 可选，PyPI 上没有）
pip install git+https://github.com/MIC-DKFZ/MedNeXt.git
```

## 快速开始（正式训练前的顺序）

```bash
# 0) 转换 AMOS22 MRI -> nnU-Net raw（胰腺标签 10 -> 1）
python convert_amos_mri_pancreas.py \
    --images_dir /path/to/amos22_mri/imagesTr \
    --labels_dir /path/to/amos22_mri/labelsTr \
    --test_images_dir /path/to/amos22_mri/imagesVa \
    --test_labels_dir /path/to/amos22_mri/labelsVa

# 1) 数据集完整性检查 + fingerprint + planning + preprocessing
nnUNetv2_plan_and_preprocess -d 150 -c 3d_fullres -verify_dataset_integrity

# 2) Exp 1 冒烟测试：只跑 fold 0，确认预处理/训练/验证/预测都正常
python run_experiment.py --exp 1 --folds 0

# 3) fold 0 正常后，跑 Exp 1 完整 5-fold
python run_experiment.py --exp 1

# 4) Exp 1 完成后跑 Exp 2（ResEnc M/L）
python run_experiment.py --exp 2

# 5) 推理（默认 5-fold 集成）
python predict.py --input nnUNet_raw/Dataset150_AMOSMRI_Pancreas/imagesTs --output predictions

# 6) 评估（Dice/Recall/Precision/HD95(mm)/ASSD(mm)）
python evaluate.py --pred_dir predictions --gt_dir nnUNet_raw/Dataset150_AMOSMRI_Pancreas/labelsTs

# 7) 实验间正式对比
python compare_experiments.py \
    --gt_dir nnUNet_raw/Dataset150_AMOSMRI_Pancreas/labelsTs \
    --pred_dirs exp1=predictions/exp1 exp2_m=predictions/exp2_m exp2_l=predictions/exp2_l
```

最后再考虑 Exp 3（间距）与 Exp 6（transformer 模型）。

## 目录结构

```
胰腺分割/
├── nnunet_utils/            # 官方包的薄封装（不改 nnU-Net 内部）
│   ├── config.py            # nnUNet_raw/preprocessed/results 路径 + 数据集/标签常量
│   ├── plans_patch.py       # 关闭深监督（官方读回字段，Exp 6 用）
│   ├── metrics.py           # Dice / Recall / Precision / HD95 / ASSD / clDice / bbox IoU
│   └── evaluation.py        # 目录级评估（evaluate.py / compare_experiments.py 共用）
├── custom_trainers/         # 官方扩展点：自定义 nnUNetTrainer 子类（Exp 6 用）
│   ├── nnUNetTrainer_UNETR.py / nnUNetTrainer_SwinUNETR.py / nnUNetTrainer_MedNeXt.py
│   └── nnUNetTrainer_DiceCEclDice.py   # Exp 5（clDice），暂存
├── experiments/             # 每个实验的编排脚本（调用官方 CLI）
├── install_custom_trainers.py
├── convert_amos_mri_pancreas.py
├── predict.py / evaluate.py / compare_experiments.py / run_experiment.py
└── requirements.txt
```

## 关键机制说明

### 官方入口全部复用
规划/预处理用 `nnUNetv2_plan_and_preprocess`，训练用 `nnUNetv2_train`，推理用
`nnUNetv2_predict`（或 `nnUNetPredictor`）。`experiments/common.py` 只负责拼命令，
环境变量 `nnUNet_raw/preprocessed/results` 指向项目内本地目录（见 `nnunet_utils/config.py`）。

### 标签：胰腺（二分类）
AMOS22 把胰腺标为器官标签 `10`；`convert_amos_mri_pancreas.py` 将 `10` 映射为 `1`，
其余（肝、肾、脾、胃等）全部归 `0`。`dataset.json` 模态写成 **MRI**。

### 训练长度：统一官方 1000 epochs
所有正式实验统一使用官方默认 1000 epochs；`run_experiment.py` 不提供 `--epochs`。

### 5-fold 交叉验证
nnU-Net v2 会自动从训练集生成标准 5-fold split。`--folds 0` 只跑 fold 0（冒烟测试），
默认跑全部 5 fold；推理默认 5-fold 集成。

### ResEnc 骨干（Exp 2）：用 planner，不是 trainer
ResEnc 架构由 **planner** 决定：`nnUNetPlannerResEncM` / `nnUNetPlannerResEncL`
规划时把残差编码器写进 plans（名为 `nnUNetResEncUNetMPlans` / `nnUNetResEncUNetLPlans`），
再用**标准 `nnUNetTrainer`** 训练。planner 的 CLI 旗标是 **`-pl`**（不是 `-planner`）。

### 间距（Exp 3）：先看自动 spacing
默认**不固定** `1×1×1 mm`。nnU-Net 会从 AMOS MRI 的真实体素间距自动决定 target spacing，
Exp 3 从生成的 plans 读回并打印该值，之后如有需要再加 `-overwrite_target_spacing`。

### 单输出 backbone（Exp 6）：关闭深监督 + 2.8.1 新接口
UNETR / SwinUNETR / MedNeXt 没有深监督头。Exp 6 用独立 plans（`nnUNetPlans_transformer`）：
1. `disable_deep_supervision` 把 plans 里的 `enable_deep_supervision` 置 `False`，同时
   trainer 的 `_build_loss` 返回不带 `DeepSupervisionWrapper` 的纯 Dice+CE；
2. `build_network_architecture` 为**实例方法**（2.8.1 接口），直接从
   `self.configuration_manager.patch_size` 读取 `img_size`。

### 评估指标（evaluate.py / compare_experiments.py）
逐类计算 **Dice / Recall / Precision / HD95 / ASSD**；HD95 与 ASSD 用 GT 的体素 spacing
换算成**毫米**。`cl_dice`（已改用新版 `skimage.morphology.skeletonize`）与 `bbox_iou`
保留在 `metrics.py`，供未来胰胆管任务使用，不参与当前胰腺评估。

## 说明

- 目标版本：**nnunetv2==2.8.1，Python 3.11**。
- planner 旗标用 `-pl`；integrity check 用 `-verify_dataset_integrity`（单横线）。
  其余旗标（`-overwrite_plans_name`、`-tr`、`-p` 等）按 nnU-Net v2 惯例。
- 开始前请先把 AMOS22 MRI 的 `imagesTr/labelsTr`（及可选的 `imagesVa/labelsVa`）路径
  传给 `convert_amos_mri_pancreas.py`。
- UNETR/SwinUNETR 对 patch size 与显存较敏感；显存不足时在对应 trainer 里调小
  `hidden_size` / `feature_size` / `depths`，或减小 patch size。

## 参考

- Isensee et al., "nnU-Net: a self-configuring method…" *Nature Methods* (2021)
- Ji et al., "AMOS: A Large-Scale Abdominal Multi-Organ Benchmark…" *arXiv* (2022)
- Hatamizadeh et al., "UNETR…" *WACV* (2022); "Swin UNETR…" *MICCAI BrainLes* (2022)
- Roy et al., "MedNeXt…" *MICCAI* (2023)
