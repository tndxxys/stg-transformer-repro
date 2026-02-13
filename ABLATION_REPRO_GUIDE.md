# STG 复现实验与消融实验指南

## 1. 目标

本指南用于说明如何：

- 在 UCI CBM 数据集上复现论文对比表与消融表；
- 一次性运行整套实验并支持断点续跑；
- 自动生成结构化结果与 Markdown 报告，便于论文/汇报撰写。


## 2. 论文必测矩阵

### 2.1 目标变量

- `kMc`
- `kMt`

### 2.2 预测步长

- `96`
- `192`
- `336`

### 2.3 评估指标

- `MSE`
- `MAE`
- `RMSE`

### 2.4 SOTA 对比模型（12个）

- `DLinear`
- `FiLM`
- `SCINet`
- `TimesNet`
- `TimeXer`
- `WPMixer`
- `iTransformer`
- `Autoformer`
- `Crossformer`
- `PatchTST`
- `DEST-GNN`
- `MA-T-GCN`

### 2.5 消融组别（4组）

- `Backbone`
- `Backbone + M1`
- `Backbone + M1 + M2`
- `Backbone + M1 + M2 + M3`（完整 STG）


## 3. 已实现文件

- 论文参考值：
  - `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/scripts/paper_reference_tables.json`
- STG 消融模型定义：
  - `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/STG_Transformer/model_ablation.py`
- STG 消融训练入口：
  - `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/STG_Transformer/train_ablation.py`
- 全量实验调度脚本：
  - `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/scripts/run_repro_ablation_suite.py`
- 论文对照汇总脚本：
  - `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/scripts/build_paper_comparison_report.py`


## 4. 消融模块固定映射

该映射已在代码中固定，并写入每次运行的 `run_meta.json`：

- `Backbone`：关闭 M1/M2/M3
- `Backbone + M1`：仅启用时间 Transformer 模块
- `Backbone + M1 + M2`：在 M1 基础上启用动态 Pearson 图
- `Backbone + M1 + M2 + M3`：在 M1+M2 基础上启用频域分解（完整 STG）


## 5. 数据与切分协议

- 数据集：`UCI CBM Dataset/uci_cbm.csv`
- 列处理：去掉 `date`
- 常量列：自动去除（默认包含 `T1`、`P1`，以及其他检测到的常量列）
- 切分方式：顺序切分，`train_ratio=0.8`、`val_ratio=0.2`


## 6. STG 最优超参数（已内置）

全量调度脚本已内置你提供的最优超参数组合：

- `kMc`：`96/192/336`
- `kMt`：`96/192/336`

正常全量运行时无需手工重复传参。


## 7. 运行方式

在仓库根目录执行：

```bash
cd ~/stg-transformer-repro
python scripts/run_repro_ablation_suite.py \
  --data_path "UCI CBM Dataset/uci_cbm.csv" \
  --run_id repro_ablation_full
```

可选参数：

- `--targets "kMc,kMt"`
- `--horizons "96,192,336"`
- `--baseline_epochs 30`
- `--baseline_patience 10`
- `--stg_epochs 30`
- `--stg_patience 15`
- `--cuda_id 0`


## 8. 输出目录结构

运行完成后，输出位于：

- `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/experiments/repro_suite/<run_id>/`

核心文件：

- `suite_meta.json`：本次运行元信息与默认参数
- `logs/*.log`：各实验日志
- `records/*.json`：各实验结构化记录
- `paper_alignment_long.csv`：逐指标长表
- `paper_alignment_wide.csv`：按论文表结构宽表
- `ablation_vs_paper.md`：可直接用于汇报的 Markdown 报告


## 9. 实验报告写作映射

建议直接从 `ablation_vs_paper.md` 取材：

- Experiment Setup（实验设置）
- Paper Values (Tables 2-5)（论文原始值）
- Reproduction Values（复现值）
- Delta Analysis（差值分析）
- Exceptions and Failures（异常与失败项）
- Conclusion and Suggested Next Steps（结论与建议）

建议写作对应关系：

- 方法部分：说明切分协议与指标计算域。
- 实验结果部分：引用 `paper_alignment_wide.csv`。
- 消融分析部分：引用表4/5对应行及 `Delta Analysis`。
- 可复现性附录：附 `suite_meta.json` 与关键日志路径。


## 10. 说明

- 调度脚本支持断点续跑：成功项会自动跳过。
- 失败项会保留状态与日志路径，便于重跑。
- 可单独重建报告（不重跑训练）：

```bash
python scripts/build_paper_comparison_report.py \
  --suite_dir experiments/repro_suite/<run_id> \
  --reference_json scripts/paper_reference_tables.json
```
