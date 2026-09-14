# nnU-Net v2 胰管/胆管分割 — 六步技术路线实验框架

基于**官方 nnU-Net v2 包**（`nnunetv2==2.8.1`，Python 3.11）的胰管 / 胆管 CT 分割实验框架。
训练 / 规划 / 预处理 / 推理全部调用官方 `nnUNetv2_*` 入口，只在官方扩展点做少量自定义
（clDice 损失、UNETR / SwinUNETR / MedNeXt 网络），**不再自建 nnU-Net 克隆**。

> 分割目标是**胰管（pancreatic duct）和胆管（bile duct）**两条细管道，不是整个胰腺；
> 两个 duct 类别是独立标签，**不合并进 pancreas**。旧的 `train.py` 与 `nnunet/`（自建实现）
> 已删除；`runs/`、`predictions/` 是旧自建实现的产物，已废弃，可删除。

## 技术路线 → 实验映射

| # | 实验 | 做法（官方机制） | 入口 |
|---|------|------------------|------|
| 1 | **基线** nnU-Net v2 3D fullres PlainConvUNet | 默认 `nnUNetTrainer`（Dice+CE，官方 1000 epochs） | [exp1_baseline.py](experiments/exp1_baseline.py) |
| 2 | **骨干对比** ResEnc M/L | planner `nnUNetPlannerResEncM/L` → `nnUNetResEncUNetM/LPlans` → 默认 `nnUNetTrainer` | [exp2_backbone.py](experiments/exp2_backbone.py) |
| 3 | **目标间距** 研究细管道 | `--overwrite_target_spacing` 生成第二套 plans | [exp3_spacing.py](experiments/exp3_spacing.py) |
| 4 | **两阶段** pancreas/肝胆 ROI → duct | 粗定位 ROI（Dataset151, 3d_lowres）→ 精分割 duct（Dataset150, 3d_fullres） | [exp4_roi.py](experiments/exp4_roi.py) |
| 5 | **连续性** Dice+CE vs +clDice | 自定义 `nnUNetTrainer_DiceCEclDice`（DS 兼容、双类别 clDice） | [exp5_cldice.py](experiments/exp5_cldice.py) |
| 6 | **其他模型** UNETR / SwinUNETR / MedNeXt | 自定义 trainer（关闭深监督、读 `configuration_manager.patch_size`） | [exp6_models.py](experiments/exp6_models.py) |

## 数据集：两个 nnU-Net dataset

| 数据集 | 标签 | 用途 |
|--------|------|------|
| `Dataset150_PancreasDuct` | `background=0, pancreatic_duct=1, bile_duct=2` | 主分割任务（Exp 1/2/3/5/6，及 Exp 4 的细阶段） |
| `Dataset151_PancreasROI` | `background=0, roi=1`（pancreas + 肝胆区） | Exp 4 的粗定位阶段 |

两者共享同一批 CT 图像（`generate_synthetic_data.py` 一次生成两套标签）。

## 安装

```bash
# Python 3.11
pip install -r requirements.txt

# MedNeXt（Exp 6 可选，PyPI 上没有）
pip install git+https://github.com/MIC-DKFZ/MedNeXt.git
```

## 快速开始

```bash
# 1) 生成合成数据（duct 数据集 + ROI 数据集，直接写入 nnU-Net raw 格式）
python generate_synthetic_data.py --num_train 20 --num_test 4

# 2) 运行某个实验（依次完成 plan + preprocess + 5-fold 训练）
python run_experiment.py --exp 1
python run_experiment.py --exp 6 --device cpu          # 或 --folds 0 1 只跑部分 fold

# 3) 推理（默认 5-fold 集成）
python predict.py --input nnUNet_raw/Dataset150_PancreasDuct/imagesTs --output predictions
python predict.py --input nnUNet_raw/Dataset150_PancreasDuct/imagesTs \
    --output predictions_c2f --coarse_to_fine          # Exp 4 两阶段

# 4) 评估（逐类 Dice/Recall/HD95(mm)/ASSD(mm)/clDice）
python evaluate.py --pred_dir predictions --gt_dir nnUNet_raw/Dataset150_PancreasDuct/labelsTs
```

## 目录结构

```
胰腺分割/
├── nnunet_utils/            # 官方包的薄封装（不改 nnU-Net 内部）
│   ├── config.py            # nnUNet_raw/preprocessed/results 路径 + 两个数据集/标签常量
│   ├── plans_patch.py       # 关闭深监督（官方读回字段）
│   └── metrics.py           # Dice / Recall / HD95 / ASSD / clDice / bbox IoU（评估用）
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

### 标签：胰管 / 胆管（双类，不合并）
`Dataset150` 的 `dataset.json` 与 `config.py` 的标签为
`{"background": 0, "pancreatic_duct": 1, "bile_duct": 2}`。合成数据里胰腺、肝、肾、脾、脊柱
作为**未标注**的背景器官保留在 CT 中，只有两条细管道被标注（duct **不**并入 pancreas）。

### 训练长度：统一官方 1000 epochs
已删除「改 plans.json 的 `num_epochs`」的做法。所有正式实验统一使用官方默认 1000 epochs；
`run_experiment.py` 不再提供 `--epochs`。

### 5-fold 交叉验证
nnU-Net v2 会自动从训练集生成标准 5-fold split。所有实验默认训练 fold 0–4，
推理默认 5-fold 集成；`run_experiment.py --folds 0 1` 可只跑部分 fold。
最后用 `evaluate.py` 对 Exp 1–6 的预测逐类比较 Dice/Recall/HD95/ASSD/clDice。

### ResEnc 骨干（Exp 2）：用 planner，不是 trainer
nnU-Net v2 里 ResEnc 架构由 **planner** 决定：`nnUNetPlannerResEncM` / `nnUNetPlannerResEncL`
规划时把残差编码器写进 plans 文件（名为 `nnUNetResEncUNetMPlans` / `nnUNetResEncUNetLPlans`），
再用**标准 `nnUNetTrainer`** 训练。没有自定义 trainer。

### 目标间距（Exp 3，真实数据重新规划）
细管道需要较细的 spacing。合成数据默认各向同性 `(1.0, 1.0, 1.0)` mm；
**真实数据请用实际体素间距重新规划**（nnU-Net 会从真实数据自动估计 spacing）。
Exp 3 用 `--overwrite_target_spacing` 生成第二套 plans 与基线对比：

```bash
# 在 experiments/exp3_spacing.py 里把 spacing 改成真实数据测得的值
nnUNetv2_plan_and_preprocess -d 150 -c 3d_fullres --verify_dataset_integrity \
    -overwrite_target_spacing 0.75 0.75 0.75 -overwrite_plans_name nnUNetPlans_fine
```

### 自定义 trainer 的安装（Exp 5/6）
`nnUNetv2_train` / `nnUNetv2_predict` 按名字在 **已安装的 `nnunetv2` 包内**查找 trainer，
因此自定义 trainer 需拷贝进该包一次（官方文档做法）：

```bash
python install_custom_trainers.py
```

Exp 5/6 脚本会在训练前自动执行这一步（幂等）。

### clDice 损失（Exp 5）：3D pooling + 深监督 target + 双类别
`nnUNetTrainer_DiceCEclDice` 包装官方的 `_build_loss`（已含深监督加权），再在**全分辨率头**上
加软骨架化 clDice。要点：
* 软腐蚀/膨胀用 `F.max_pool3d`（3D pooling），kernel 为 3 元组 `(3,1,1)`/`(1,3,1)`/`(1,1,3)`，
  只作用在三个空间维；
* 深监督开启时 nnU-Net 把输出和 target 都传成 list，这里取 `net_output[0]` / `target[0]`
  的全分辨率元素计算 clDice；
* 对每个前景类别（pancreatic_duct、bile_duct）各算一个 clDice 再相加。

### 单输出 backbone（Exp 6）：关闭深监督 + 2.8.1 新接口
UNETR / SwinUNETR / MedNeXt 没有深监督头。Exp 6 用独立 plans
（`nnUNetPlans_transformer`）并做两件事：
1. `disable_deep_supervision` 把 plans 里的 `enable_deep_supervision` 置 `False`，同时
   trainer 的 `_build_loss` 返回**不带** `DeepSupervisionWrapper` 的纯 Dice+CE，保证数据加载的
   单一 target、网络、损失三者一致；
2. `build_network_architecture` 为**实例方法**（2.8.1 接口），直接从
   `self.configuration_manager.patch_size` 读取 `img_size`（不再注入 arch_init_kwargs）。

### 两阶段 ROI → duct（Exp 4）
粗阶段在 `Dataset151` 上训练 `3d_lowres`，定位 pancreas + 肝胆 ROI；细阶段在 `Dataset150`
上训练 `3d_fullres`，在 ROI 内分割 duct。`predict.py --coarse_to_fine` 串起两阶段：
粗预测 ROI → 裁剪（含 margin）→ 细分割 duct → 贴回全图。

### 评估指标（evaluate.py）
逐类计算 Dice / Recall / HD95 / ASSD / clDice；HD95 与 ASSD 用 GT 的体素 spacing
换算成**毫米**。另提供 `--localization`（bbox IoU）用于粗定位阶段的快速检查。

## 说明

- 目标版本：**nnunetv2==2.8.1，Python 3.11**（`--overwrite_target_spacing` /
  `-planner` / 实例方法 `build_network_architecture` 均可用）。
- 各 CLI 旗标按 nnU-Net v2 惯例写成单横线（`-overwrite_plans_name`、`-planner`、`-tr`、`-p` 等）；
  个别版本若不同，先 `nnUNetv2_plan_and_preprocess -h` 核对。
- UNETR/SwinUNETR 对 patch size 与显存较敏感；显存不足时在对应 trainer 里调小
  `hidden_size` / `feature_size` / `depths`，或减小 patch size。

## 参考

- Isensee et al., "nnU-Net: a self-configuring method…" *Nature Methods* (2021)
- Shit et al., "clDice — a novel topology-preserving loss…" *CVPR* (2021)
- Hatamizadeh et al., "UNETR…" *WACV* (2022); "Swin UNETR…" *MICCAI BrainLes* (2022)
- Roy et al., "MedNeXt…" *MICCAI* (2023)
