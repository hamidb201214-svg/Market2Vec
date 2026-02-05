# Market2Vec
## Trademark-Based Product Timeline Embeddings (Forecasting MLM)

This repo builds and trains **Market2Vec** from **trademark data** using a **sequence-of-products** view:

- **Firms (owners)** are treated as entities with a timeline of **products**
- Each **product** is a **sequential event**
- Each product has an **item set** (goods/services descriptors) treated as a **basket** (set, not order)

---

## Two training versions (two objectives)

We support **two versions** of the MLM objective:

### Version A — Forecasting (last-event prediction)
Purpose: learn to **forecast the last product’s items** from the firm’s earlier product history.

- We identify the **last product event** in the sequence: the segment between the last `[APP]` and the next `[APP_END]` (or `[SEP]`)
- We **force-mask `ITEM_*` tokens inside that last event** (mask probability = 1.0)
- (Optional) random masking elsewhere can be turned off for “clean forecasting” evaluation

This version is best when your downstream use-case is “given past trademark products, predict items in the most recent/next product”.

### Version B — Random MLM over the full product sequence
Purpose: learn general co-occurrence/semantic structure of items in firm timelines (classic MLM).

- We mask tokens **randomly across the whole sequence** with probability `p` (e.g., 15%)
- This includes items across **all product events**, not only the last one
- This version behaves like standard BERT MLM, but applied to your product timeline format

This version is best when you want broad embeddings capturing item relationships and temporal context without specifically focusing on forecasting the last event.

---

## How to enable each version in code

The behavior is controlled by the masking probabilities used in the collator:

- `TRAIN_RANDOM_MLM_PROB`
- `EVAL_RANDOM_MLM_PROB`

And by whether you “force-mask last event items” (enabled in the forecasting collator logic).

### Recommended settings

#### Forecasting-only (Version A)
- Train: `TRAIN_RANDOM_MLM_PROB = 0.0` (no random MLM noise)
- Eval:  `EVAL_RANDOM_MLM_PROB  = 0.0`
- Force-masking last event `ITEM_*` stays **ON**

This focuses learning and evaluation on last-event item prediction.

#### Forecasting + regularization (Version A + random noise)
- Train: `TRAIN_RANDOM_MLM_PROB = 0.15`
- Eval:  `EVAL_RANDOM_MLM_PROB  = 0.0`
- Force-masking last event `ITEM_*` stays **ON**

This is the default “forecasting twist” setup: train with extra random MLM, evaluate cleanly on forecasting.

#### Random MLM across full sequence (Version B)
- Train: `TRAIN_RANDOM_MLM_PROB = 0.15`
- Eval:  `EVAL_RANDOM_MLM_PROB  = 0.15` (or any non-zero)
- (Optional) disable force-masking last-event items if you want *pure* standard MLM

> Note: In the current `ForecastingCollator`, force-masking last-event items is always applied.  
> If you want **pure random MLM** (no forecasting), add a flag like `force_last_event=False` and skip the `prob[force_mask] = 1.0` step.

---

## What the forecasting masking means (in practice)

A packed firm sequence looks like:

[CLS]
DATE_YYYY_MM [APP] NICE_* ... ITEM_* ... [APP_END]
DATE_YYYY_MM [APP] NICE_* ... ITEM_* ... [APP_END]
...
DATE_YYYY_MM [APP] NICE_* ... ITEM_* ... [APP_END] <-- last event
[SEP]


- **Version A:** masks `ITEM_*` in the last `[APP]..[APP_END]` segment (forecasting target)
- **Version B:** masks tokens randomly across the entire sequence (classic MLM)

---

## Metrics

Validation reports:
- **AccAll**: accuracy over all masked tokens
- **Item@K**: Top-K accuracy restricted to masked positions where the true label is an `ITEM_*` token

For forecasting, **Item@K** is the main metric because it directly measures how well the model predicts items in the last product basket.

---
## Results — MarketBERT (pretrained Market2Vec checkpoint)

### Training Summary
- **Model:** `A4_full_fixed_alpha_optionA_h512_h32`
- **Best validation loss:** `3.6433`

### Validation (HARD)
- **Acc@1:** `0.5996`
- **Acc@5:** `0.6651`
- **Acc@10:** `0.6944`

> “HARD” refers to the stricter evaluation setting used in our validation protocol (forecasting-focused metrics on masked targets).

---

## Usage (Hugging Face)

```python
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("HamidBekam/MarketBERT")
model = AutoModel.from_pretrained("HamidBekam/MarketBERT")
