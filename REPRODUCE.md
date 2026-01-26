# 论文复现步骤（含每步说明）

本文档以当前项目为基础，给出复现实验的完整流程，并解释每一步的作用。默认使用 UCI CBM 数据集（`UCI CBM Dataset`）。

注意：论文表 1 的变量描述与 UCI CBM 数据集有版本差异（本数据为 16 个输入 + 2 个目标），且原论文未公开全部对比模型代码/超参，因此这里提供的是“可复现实验流程”，而不是保证完全复现论文表格数值。

---

## 1. 准备环境

目的：确保 Python 与依赖库一致，避免运行报错。

推荐使用 conda 环境（可选）：

```bash
conda create -n stg python=3.9 -y
conda activate stg
```

安装依赖：

```bash
pip install -r requirements.txt
```

基线模型依赖各自仓库的 `requirements.txt`（建议在同一环境中安装，避免冲突）：

```bash
pip install -r baselines/Time-Series-Library/requirements.txt
pip install -r baselines/TimeXer/requirements.txt
pip install -r baselines/WPMixer/requirements.txt
pip install -r baselines/DeSTGNN_full/requirements.txt
pip install -r baselines/matgcn/requirements.txt
```

说明：如果某个基线缺少 `requirements.txt`，请按其 README 中的依赖说明安装。

---

## 2. 生成统一 CSV 数据

目的：Time-Series-Library/TimeXer/WPMixer 需要带 `date` 列的 CSV，UCI CBM 原始数据是 `data.txt`。

执行：

```bash
python scripts/prepare_uci_cbm.py \
  --data_path "UCI CBM Dataset/data.txt" \
  --features_path "UCI CBM Dataset/Features.txt" \
  --output "UCI CBM Dataset/uci_cbm.csv"
```

说明：
- 脚本会自动读 `Features.txt`，生成英文列名（包含 `kMc` 和 `kMt`）。
- `date` 列是基于行号生成的等间隔时间戳，主要用于基线模型的时间特征编码。

---

## 3. 运行 STG-Transformer（本文模型）

目的：在本项目内复现论文提出的 STG-Transformer 预测结果。

建议分别对 `kMc` 与 `kMt` 运行（论文也分别报告两者指标）。

示例（预测步长 96）：

```bash
 python -m STG_Transformer.train \
    --data_path "UCI CBM Dataset/uci_cbm.csv" \
    --seq_len 96 \
    --pred_len 96 \
    --target_cols "kMc" \
    --drop_cols "date" \

```

切换为 `kMt`：

```bash
python -m STG_Transformer.train \
  --data_path "UCI CBM Dataset/uci_cbm.csv" \
  --seq_len 96 \
  --pred_len 96 \
  --target_cols "kMt" \
  --drop_cols "date"
```

说明：
- `seq_len/pred_len` 可改为 192 或 336，对应论文的预测长度设置。
- 默认使用动态图（皮尔逊相关）+ FFT 分解 + 双通道编码器/解码器。
- 评估指标为 MSE/MAE/RMSE（在验证集上逆标准化后计算）。

---

## 4. 运行时间序列基线模型（8 个）

目的：复现实验中的 8 个时序基线（DLinear/FiLM/SCINet/TimesNet/iTransformer/Autoformer/Crossformer/PatchTST）。

这些模型在 `baselines/Time-Series-Library` 中，使用 `custom` 数据集模式。

示例（DLinear，kMc）：

```bash
python baselines/Time-Series-Library/run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id cbm_DLinear_kMc_96 --model DLinear \
  --data custom --root_path "UCI CBM Dataset" --data_path "uci_cbm.csv" \
  --features MS --target kMc --freq t \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 18 --dec_in 18 --c_out 1
```

说明：
- `features=MS` 表示多变量输入、单变量输出。
- `enc_in/dec_in=18`，因为包含 16 个输入变量 + kMc + kMt 共 18 列（不含 date）。
- 其余模型只需替换 `--model`。

---

## 5. 运行 TimeXer 与 WPMixer

### TimeXer

```bash
python baselines/TimeXer/run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id cbm_TimeXer_kMc_96 --model TimeXer \
  --data custom --root_path "UCI CBM Dataset" --data_path "uci_cbm.csv" \
  --features MS --target kMc --freq t \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 18 --dec_in 18 --c_out 1
```

### WPMixer

```bash
python baselines/WPMixer/run_LTF.py \
  --data custom --root_path "UCI CBM Dataset" --data_path "uci_cbm.csv" \
  --features MS --target kMc --freq t \
  --c_in 18 --c_out 1 \
  --seq_len 96 --pred_len 96
```

说明：
- WPMixer 已修改为支持 `custom`。
- 两个模型同样需要对 `kMt` 单独跑一遍。

---

## 6. 运行图模型基线（DEST-GNN / MA-T-GCN）

目的：复现论文中的图模型对比基线。

### 6.1 生成图基线输入与配置

```bash
python scripts/prepare_graph_baselines.py \
  --csv_path "UCI CBM Dataset/uci_cbm.csv" \
  --pred_len 96
```

说明：
- 该脚本会生成：
  - DeSTGNN 的 `cbm.npz` + `cbm_adj.npy` + `CBM.conf`
  - matgcn 的 `cbm.npz` + `cbm_distance.csv` + `cbm.json`
- 邻接矩阵默认按变量相关性构图（可选 `--adj_mode full|corr|topk`）。

### 6.2 运行 DeSTGNN

```bash
python baselines/DeSTGNN_full/prepareData.py \
  --config baselines/DeSTGNN_full/configurations/CBM.conf

python baselines/DeSTGNN_full/train_DeSTGNN.py \
  --config baselines/DeSTGNN_full/configurations/CBM.conf --cuda 0
```

### 6.3 运行 MA-T-GCN（matgcn）

```bash
python baselines/matgcn/main.py baselines/matgcn/config/cbm.json
```

说明：
- matgcn 由于默认包含 7 天周期特征，可能导致样本不足；当前适配默认使用小时步长 `[1,2,3]`。
- 若改为 192/336，请重新运行 `prepare_graph_baselines.py --pred_len 192/336`。

---

## 7. 对齐论文指标与对比

目的：对比不同模型在 kMc/kMt 的 MSE/MAE/RMSE。

建议操作：
1) 对每个模型分别运行 kMc 与 kMt。
2) 在 96/192/336 三个预测长度下重复实验。
3) 统一记录验证/测试集指标，整理成表格对比。

---

## 8. 已知差异与注意事项

- 数据集与论文表 1 有差异（无 date 列，变量数量不同）。
- 时间戳为人工生成（用于时间特征编码），不代表真实采样时间。
- 图模型基线使用变量相关性构图，而非论文中的真实物理拓扑。

这些差异会影响与论文原表格数值的完全一致性，但可用于复现“方法流程 + 相对对比趋势”。

