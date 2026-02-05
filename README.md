# Market2Vec — Trademark-Based Product Timeline Embeddings (Forecasting MLM)

This repo builds and trains **Market2Vec** from **trademark data** using a **sequence-of-products** view:

- **Firms (owners)** are treated as entities with a timeline of **products**
- Each **product** is a **sequential event**
- Each product has an **item set** (goods/services descriptors) treated as a **basket** (set, not order)

Training uses a BERT-style **Masked Language Modeling (MLM)** objective with a **forecasting twist**:
- **Always mask `ITEM_*` tokens in the last product event** so the model learns to **forecast product items** from the firm’s prior timeline context.
- Optionally apply standard random MLM masking during training for regularization.

---

## Repository contents

- `preprocess.py` (your preprocessing script)
- `train.py` (your training script using `ForecastingCollator`)
- `README.md`

---

## 1) Preprocessing approach

### Goal
Convert raw trademark rows into:
1) **Event-level product records** per firm
2) **Packed sequences** of events (windows) with `MAX_LEN=512`
3) A compact vocabulary where head items cover **95% of item occurrences**

### Input
The preprocessing script reads a Parquet file:

- `PARQUET_PATH = /.../1996_2025_TM_sequence_with_labelid.parquet`

Required columns:

- `owner_id`
- `ApplicationNumber`
- `Nice`
- `RegistrationDate`
- `Year`
- `full_description_norm`

`full_description_norm` is expected to be a semicolon-separated string of item descriptors.

---

## 2) Vocabulary design (95% item coverage)

To keep the vocab compact and robust:

✅ **Head vocabulary by coverage**
- Explode all items (one row per item occurrence)
- Count item frequencies
- Keep the most frequent items that cumulatively cover `ITEM_COVERAGE = 0.95` of *all occurrences*
- All remaining tail items map to **`ITEM_UNK`**
- Head items are remapped into a compact id space: `ITEM_0 ... ITEM_{H-1}`
  - `ITEM_0` is the most frequent head item

This makes the model focus capacity on common items while still representing rare items via `ITEM_UNK`.

The saved pickle includes:
- `id2item_head`: list mapping `new_id -> original item string`
- `head_old_item_ids`: mapping from `new_id -> old factorized id`
- `item_coverage`: the coverage threshold used

---

## 3) Event construction (products as events, baskets as sets)

One **event** = one trademark application:

- Group by `(owner_id, ApplicationNumber)`
- Event timestamp:
  - use `RegistrationDate` if available
  - else fallback to `Year-01-01`

Within each event:
- NICE classes are treated as a **set**:
  - dedupe + sort + cap (`MAX_NICE_PER_APP`)
- Items are treated as a **set**:
  - map to compact head ids or UNK (`-1`)
  - dedupe + canonical ordering + cap (`MAX_ITEMS_PER_APP`)
  - `ITEM_UNK` appears at most once and is kept even when capped

So each event becomes a clean basket:
DATE_YYYY_MM [APP] NICE_... ITEM_... [APP_END]

---

## 4) Tokenization schema

The preprocessing writes a custom `vocab.txt` and builds a `BertTokenizer` from it.

Vocabulary includes:

**Special/event markers**
- `[PAD] [UNK] [CLS] [SEP] [MASK] [APP] [APP_END]`

**Time**
- `DATE_YYYY_MM` for every month in the dataset date range (month-start aligned)

**NICE**
- `NICE_01 ... NICE_45`
- `NICE_UNK`

**Items**
- `ITEM_UNK`
- `ITEM_0 ... ITEM_{H-1}` (head items only)

---

## 5) Sequence building (MAX_LEN=512, never cut inside an event)

The preprocessing builds **event-packed windows** per firm:

- Final sequence length `<= 512` including `[CLS]` and `[SEP]`
- Uses `MAX_TL = 510` for payload tokens
- **Never cuts inside an event**: it packs full events until the next event would exceed length.
- Generates multiple windows by using event-level start offsets:
  - `START_EVENT_OFFSETS = (0, 1, 2)`
- Windows shorter than `MIN_SEQ_TOKENS` are discarded.

Each output sequence is:

[CLS] (DATE + [APP] + NICE-set + ITEM-set + [APP_END]) * K [SEP]

---

## 6) Train/valid/test splitting (month-based by last event)

Each packed window is assigned to a split based on the **month of its last event**:

- Train: `last_ym <= TRAIN_CUTOFF_YM` (default `202312`)
- Valid: `VALID_START_YM..VALID_END_YM` (default `202401..202412`)
- Test:  `last_ym >= TEST_START_YM` (default `202501+`)

This preserves temporal ordering and avoids leaking future products into training.

---

## 7) Set-of-items behavior (optional shuffle in TRAIN)

Within an event, NICE and Items are **sets**. To avoid accidental learning of token order, TRAIN windows can apply **deterministic shuffling**:

- `TRAIN_SHUFFLE_ITEMS = True` (default)
- `TRAIN_SHUFFLE_NICE = False` (default)
- Shuffling is deterministic per window using a stable hash seed (`stable_seed(...)`)

This keeps the “basket” assumption while still producing varied token order during training.

---

## 8) Preprocessing output

The script writes:

`OUTPUT_DIR/processed_data.pkl`

with keys:

- `train`, `valid`, `test`: lists of sequences (list[int])
- `vocab_file`: path to generated vocab.txt
- `id2item_head`, `head_old_item_ids`
- `item_coverage`
- `max_len`

