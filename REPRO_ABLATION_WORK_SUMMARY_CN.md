# 复现实验与消融自动化改造工作总结（中文）

## 一、工作目标

本次改造目标是把 STG 论文复现实验从“手工逐个跑”升级为“可批量、可追溯、可对照论文”的自动化流程，覆盖：

- 论文 SOTA 对比（表2/3）；
- 论文消融实验（表4/5）；
- 自动产出结构化结果与汇报文档。


## 二、核心新增能力

### 1) 论文对照基准库

新增文件：

- `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/scripts/paper_reference_tables.json`

作用：

- 固化论文表2/3/4/5中的指标数值；
- 支持按 `group(target/table) + horizon + model + metric` 逐项对照。


### 2) 消融模型独立实现（不破坏原训练入口）

新增文件：

- `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/STG_Transformer/model_ablation.py`

新增模型：

- `STGAblationModel`

支持模块开关：

- `use_m1`: 时间 Transformer 编解码
- `use_m2`: 动态 Pearson 图构建
- `use_m3`: 频域分解（低/高频双路）

对应论文消融映射：

- `Backbone` → `use_m1/use_m2/use_m3 = False/False/False`
- `Backbone + M1` → `True/False/False`
- `Backbone + M1 + M2` → `True/True/False`
- `Backbone + M1 + M2 + M3` → `True/True/True`


### 3) 消融训练入口

新增文件：

- `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/STG_Transformer/train_ablation.py`

特性：

- 独立参数 `--ablation_mode` 控制 4 组消融；
- 保留与主训练一致的数据流程（含标准化与反标准化评估）；
- 每次运行输出统一工件：
  - `metrics.json`
  - `pred.npy`
  - `true.npy`
  - `pred_std.npy`
  - `true_std.npy`
  - `history.json`
  - `run_meta.json`
  - `best_model.pth`


### 4) 全量实验调度器

新增文件：

- `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/scripts/run_repro_ablation_suite.py`

主要功能：

- 自动构造实验矩阵（2目标 × 3步长 × 4消融 + 12基线）；
- 执行并记录每个实验命令与日志；
- 断点续跑（已成功任务自动跳过）；
- 统一结果落盘到 `records/*.json`；
- 结束后自动调用报告脚本。

已内置你给定的 STG 最优参数：

- `kMc`: 96/192/336
- `kMt`: 96/192/336


### 5) 论文对照报告生成器

新增文件：

- `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/scripts/build_paper_comparison_report.py`

输出：

- `paper_alignment_long.csv`（逐项长表）
- `paper_alignment_wide.csv`（按表结构宽表）
- `ablation_vs_paper.md`（可直接用于汇报）

对照字段包含：

- 论文值（原始精度 + 3位小数）
- 复现值（原始精度 + 3位小数）
- 绝对差值
- 相对误差
- 是否优于论文
- 状态/日志路径（失败项可追溯）


## 三、基线兼容性改造

### 1) WPMixer 统一导出

修改文件：

- `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/baselines/WPMixer/exp/exp_main.py`

新增导出：

- `metrics.npy`
- `pred.npy`
- `true.npy`


### 2) MA-T-GCN 最优轮次导出

修改文件：

- `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/baselines/matgcn/tools/trainer.py`

新增导出：

- `best_pred.npy`
- `best_true.npy`
- `best_metrics.json`


### 3) 图模型数据准备增强

修改文件：

- `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/scripts/prepare_graph_baselines.py`

新增行为：

- 默认自动去除常量列（含 `T1/P1`）；
- 可通过参数保留常量列。


## 四、关键问题修复

在调度器中修复了图模型目标提取逻辑：

- 文件：`/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/scripts/run_repro_ablation_suite.py`
- 修复点：`DEST-GNN` / `MA-T-GCN` 的 `kMc/kMt` 目标索引提取维度。
- 原风险：按“目标数量”取轴，可能错取通道。
- 现实现：按完整特征维数取目标索引，确保 `kMc/kMt` 指标对应正确。


## 五、工程保证

- 原入口 `STG_Transformer/train.py` 未被破坏；
- 新增能力均在独立脚本/文件中实现；
- 所有新增主脚本已通过 `python3 -m py_compile` 语法检查；
- 报告脚本可独立运行并稳定生成三类汇总文件。


## 六、如何执行（建议）

### 1) 全量执行

```bash
cd ~/stg-transformer-repro
python scripts/run_repro_ablation_suite.py \
  --data_path "UCI CBM Dataset/uci_cbm.csv" \
  --run_id repro_ablation_full
```

### 2) 仅重建报告

```bash
python scripts/build_paper_comparison_report.py \
  --suite_dir experiments/repro_suite/repro_ablation_full \
  --reference_json scripts/paper_reference_tables.json
```


## 七、输出目录

- `/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/experiments/repro_suite/<run_id>/`

主要文件：

- `suite_meta.json`
- `records/*.json`
- `logs/*.log`
- `paper_alignment_long.csv`
- `paper_alignment_wide.csv`
- `ablation_vs_paper.md`


## 八、可直接用于汇报的结论模板（建议）

1. 说明实验矩阵覆盖了论文表2/3/4/5。
2. 指出所有实验均可追溯到命令、日志与元数据。
3. 逐表呈现论文值与复现值差异，并解释偏差来源。
4. 对失败项给出复跑计划与日志定位路径。
5. 给出下一轮优化方向（如统一硬件、统一随机种子、多次重复取均值）。
