# nnU-Net v2 胰腺分割 — 六步技术路线实验框架

基于**官方 nnU-Net v2 包**（`nnunetv2`）的胰腺 CT 分割实验框架。训练/规划/预处理/推理全部调用
官方 `nnUNetv2_*` 入口，只在官方扩展点上做少量自定义（clDice 损失、UNETR/SwinUNETR/MedNeXt 网络），
**不再自建 nnU-Net 克隆**。

## 技术路线 → 实验映射

| # | 实验 | 做法（官方机制） | 入口 |
|---|------|------------------|------|
| 1 | **基线** nnU-Net v2 3D fullres PlainConvUNet | 默认 `nnUNetTrainer`（Dice+CE） | [exp1_baseline.py](experiments/exp1_baseline.py) |
| 2 | **骨干对比** ResEnc M/L | 官方 `nnUNetTrainerResEncM` / `nnUNetTrainerResEncL` | [exp2_backbone.py](experiments/exp2_backbone.py) |
| 3 | **目标间距** 研究微细管道 | `--overwrite_target_spacing` 生成第二套 plans | [exp3_spacing.py](experiments/exp3_spacing.py) |
| 4 | **ROI / coarse-to-fine** 类别不平衡 | `3d_lowres` 定位 + `3d_fullres` 精分割 | [exp4_roi.py](experiments/exp4_roi.py) |
| 5 | **连续性** Dice+CE vs +clDice | 自定义 `nnUNetTrainer_DiceCEclDice` | [exp5_cldice.py](experiments/exp5_cldice.py) |
| 6 | **其他模型** UNETR / SwinUNETR / MedNeXt | 自定义 trainer（MONAI / MedNeXt） | [exp6_models.py](experiments/exp6_models.py) |

## 安装

```bash
pip install -r requirements.txt

# MedNeXt（Exp 6 可选，PyPI 上没有）
pip install git+https://github.com/MIC-DKFZ/MedNeXt.git
```

## 快速开始

```bash
# 1) 生成合成数据（直接写入 nnU-Net raw 格式；--add_duct 生成细导管用于 Exp 3/5）
python generate_synthetic_data.py --num_train 20 --num_test 4 --add_duct

# 2) 运行某个实验（会依次完成 plan + preprocess + train）
python run_experiment.py --exp 1
python run_experiment.py --exp 6 --device cuda --epochs 500

# 3) 推理
python predict.py --input nnUNet_raw/Dataset150_PancreasCT/imagesTs --output predictions
python predict.py --input nnUNet_raw/Dataset150_PancreasCT/imagesTs \
    --output predictions_c2f --coarse_to_fine        # Exp 4 两阶段

# 4) 评估（Dice / HD95 / clDice）
python evaluate.py --pred_dir predictions --gt_dir nnUNet_raw/Dataset150_PancreasCT/labelsTs
```

## 目录结构

```
胰腺分割/
├── nnunet_utils/            # 官方包的薄封装（不改 nnU-Net 内部）
│   ├── config.py            # nnUNet_raw/preprocessed/results 路径 + 数据集常量
│   ├── plans_patch.py       # 编辑 plans 文件的 num_epochs / img_size（官方读回字段）
│   └── metrics.py           # Dice / HD95 / clDice（最终评估用）
├── custom_trainers/         # 官方扩展点：自定义 nnUNetTrainer 子类
│   ├── nnUNetTrainer_DiceCEclDice.py
│   ├── nnUNetTrainer_UNETR.py / nnUNetTrainer_SwinUNETR.py / nnUNetTrainer_MedNeXt.py
├── experiments/             # 每个实验的编排脚本（调用官方 CLI）
├── install_custom_trainers.py
├── generate_synthetic_data.py
├── predict.py / evaluate.py / run_experiment.py
└── requirements.txt
```

## 关键机制说明

### 官方入口全部复用
规划/预处理用 `nnUNetv2_plan_and_preprocess`，训练用 `nnUNetv2_train`，推理用
`nnUNetv2_predict`（或 `nnUNetPredictor`）。`experiments/common.py` 只负责拼命令，
环境变量 `nnUNet_raw/preprocessed/results` 指向项目内本地目录（见 `nnunet_utils/config.py`）。

### 自定义 trainer 的安装（Exp 5/6）
`nnUNetv2_train` / `nnUNetv2_predict` 按名字在 **已安装的 `nnunetv2` 包内**查找 trainer。
因此自定义 trainer 需拷贝进该包一次（官方文档的做法）：

```bash
python install_custom_trainers.py
```

Exp 5/6 的脚本会在训练前自动执行这一步（幂等）。

### 自定义网络的输入尺寸（Exp 6）
UNETR / SwinUNETR 需要固定 `img_size == patch_size`。Exp 6 用独立的 plans
（`nnUNetPlans_transformer`），并把 `patch_size` 注入 `arch_init_kwargs`（`nnunet_utils/plans_patch.py`）。
若你的 nnU-Net 版本从别的字段读取 `arch_init_kwargs`，改 `plans_patch.patch_arch_init_kwargs`
里写入的 key 即可（一处）。

### 目标间距（Exp 3）
`nnUNetv2_plan_and_preprocess --overwrite_target_spacing 1.0 1.0 1.0 --overwrite_plans_name nnUNetPlans_1mm`
生成第二套 plans 与预处理数据，与基线 `nnUNetPlans` 互不影响。

## 说明

- 目标 nnU-Net 版本：**≥ 2.2**（`--overwrite_target_spacing` / `--overwrite_plans_name` /
  6 参数 `build_network_architecture` 静态方法均在该版本起可用）。
- `runs/`、`predictions/` 是旧自建实现的产物，已废弃，可删除。
- UNETR/SwinUNETR 对 patch size 与显存较敏感；若显存不足，在对应 trainer 里调小
  `hidden_size` / `feature_size` / `depths` 或减小 patch size。

## 参考

- Isensee et al., "nnU-Net: a self-configuring method…" *Nature Methods* (2021)
- Shit et al., "clDice — a novel topology-preserving loss…" *CVPR* (2021)
- Hatamizadeh et al., "UNETR…" *WACV* (2022); "Swin UNETR…" *MICCAI BrainLes* (2022)
- Roy et al., "MedNeXt…" *MICCAI* (2023)
