# Baseline Runs (UCI CBM Dataset)

These commands assume you generated `UCI CBM Dataset/uci_cbm.csv` using:

```bash
python scripts/prepare_uci_cbm.py \
    --data_path "UCI CBM Dataset/data.txt" \
    --features_path "UCI CBM Dataset/Features.txt" \
    --output "UCI CBM Dataset/uci_cbm.csv"
```

Common values:
- `ROOT="UCI CBM Dataset"`
- `DATA="uci_cbm.csv"`
- `VARS=18` (all variables except `date`)
- `SEQ=96`, `LABEL=48`, `PRED=96` (replace with 192/336 as needed)
- `FREQ=t` (minute-level synthetic timestamps)

## Time-Series-Library baselines

Models: `DLinear`, `FiLM`, `SCINet`, `TimesNet`, `iTransformer`, `Autoformer`, `Crossformer`, `PatchTST`

Example (kMc target):

```bash
python baselines/Time-Series-Library/run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id cbm_DLinear_kMc_96 --model DLinear \
  --data custom --root_path "UCI CBM Dataset" --data_path "uci_cbm.csv" \
  --features MS --target kMc --freq t \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 18 --dec_in 18 --c_out 1 \
  --train_epochs 10 --batch_size 32
```

Repeat with `--target kMt` and adjust `--model` for each baseline.

## TimeXer

```bash
python baselines/TimeXer/run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id cbm_TimeXer_kMc_96 --model TimeXer \
  --data custom --root_path "UCI CBM Dataset" --data_path "uci_cbm.csv" \
  --features MS --target kMc --freq t \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 18 --dec_in 18 --c_out 1 \
  --train_epochs 10 --batch_size 32
```

## WPMixer

```bash
python baselines/WPMixer/run_LTF.py \
  --data custom --root_path "UCI CBM Dataset" --data_path "uci_cbm.csv" \
  --features MS --target kMc --freq t \
  --c_in 18 --c_out 1 \
  --seq_len 96 --pred_len 96 \
  --train_epochs 10 --batch_size 128
```

## Graph baselines (DEST-GNN / MA-T-GCN)

First prepare graph inputs/configs:

```bash
python scripts/prepare_graph_baselines.py \
  --csv_path "UCI CBM Dataset/uci_cbm.csv" \
  --pred_len 96
```

### DeSTGNN

```bash
python baselines/DeSTGNN_full/prepareData.py \
  --config baselines/DeSTGNN_full/configurations/CBM.conf

python baselines/DeSTGNN_full/train_DeSTGNN.py \
  --config baselines/DeSTGNN_full/configurations/CBM.conf --cuda 0
```

### MA-T-GCN (matgcn)

```bash
python baselines/matgcn/main.py baselines/matgcn/config/cbm.json
```
