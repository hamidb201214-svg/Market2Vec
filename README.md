# Forecasting MLM (BERT) — Custom Collator Training Script

This repo trains a **BERT-style Masked Language Model** (`BertForMaskedLM`) on **pre-tokenized sequences** stored in a pickle file, using a **custom “forecasting” masking strategy**:

- (Train) Standard random MLM masking (e.g., 15%)
- (Train + Eval) **Force-mask ITEM tokens in the last event segment** (between the last `[APP]` and `[APP_END]` / `[SEP]`)
- Reports **overall masked accuracy** and **ITEM-only Top-K accuracy** (Item@1/5/10)

---

## What the code does

### Data flow
1. Load `processed_data.pkl`
2. Build a `BertTokenizer` from `data["vocab_file"]`
3. Wrap `data["train"]` / `data["valid"]` into a PyTorch `Dataset`
4. Use `ForecastingCollator` to create `input_ids`, `attention_mask`, `labels`
5. Train `BertForMaskedLM` with AdamW + linear warmup schedule
6. Evaluate per epoch on validation set (forecasting-only masking by default)
7. Save checkpoints (`best_model`, `latest_model`) + tokenizer

### ForecastingCollator (key idea)
The collator converts each batch into an MLM objective:
- Builds `labels` from the original tokens
- Chooses mask positions:
  - random masking probability (train) or none (eval)
  - **always masks** `ITEM_*` tokens in the last event segment
- Sets non-masked labels to `-100` (ignored by loss)
- Replaces 80% of masked tokens with `[MASK]`

---

## Requirements

- Python 3.9+ (recommended)
- PyTorch
- Transformers
