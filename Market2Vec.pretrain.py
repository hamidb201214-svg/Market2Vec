import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

# ============================================================
# 0) CONFIG
# ============================================================
PICKLE_PATH = "/home/ubuntu/storage_b/data/MarketSBERTa/output_granular_items_eval_month/processed_data.pkl"
OUTPUT_DIR  = "/home/ubuntu/storage_b/data/MarketSBERTa/output_granular_items_eval_month/model_ckpts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 256

# ---- Model (Architecture) ----
HIDDEN_SIZE = 256
NUM_LAYERS  = 6
NUM_HEADS   = 8

# ---- Training ----
BATCH_SIZE    = 64
EPOCHS        = 10
LR            = 1e-4
WEIGHT_DECAY  = 0.0
WARMUP_RATIO  = 0.1
GRAD_CLIP     = 1.0

# ---- Masking ----
TRAIN_RANDOM_MLM_PROB = 0.15
EVAL_RANDOM_MLM_PROB  = 0.0
MASK_NICE_IN_LAST_EVENT = False

TOPK_LIST = (1, 5, 10)
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1) DATASET + DATALOADER
# ============================================================
class SeqIdsDataset(Dataset):
    """Takes pre-tokenized sequences (list[int]) and pads/truncates to MAX_LEN."""
    def __init__(self, sequences, max_len, pad_id, sep_id):
        self.seqs = sequences
        self.max_len = max_len
        self.pad_id = pad_id
        self.sep_id = sep_id

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = self.seqs[idx]

        # truncate
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            ids[-1] = self.sep_id

        attn = [1] * len(ids)
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids  = ids + [self.pad_id] * pad_len
            attn = attn + [0] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class ForecastingCollator:
    """
    Random MLM (train) + FORCE mask ITEM tokens in last [APP]..[APP_END] segment.
    Eval uses random_mlm_prob=0.0 -> forecasting-only.
    """
    def __init__(self, tokenizer: BertTokenizer, random_mlm_prob: float, mask_nice: bool):
        self.tok2id = tokenizer.vocab
        self.vocab_size = tokenizer.vocab_size

        self.app_id     = self.tok2id["[APP]"]
        self.app_end_id = self.tok2id["[APP_END]"]

        self.pad_id  = tokenizer.pad_token_id
        self.cls_id  = tokenizer.cls_token_id
        self.sep_id  = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id

        self.random_mlm_prob = float(random_mlm_prob)
        self.mask_nice = bool(mask_nice)

        # Precompute vocab masks: ITEM_* and NICE_*
        item_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        nice_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for tok, tid in self.tok2id.items():
            if tok.startswith("ITEM_"):
                item_mask[tid] = True
            elif tok.startswith("NICE_"):
                nice_mask[tid] = True

        self.item_mask_vocab = item_mask
        self.nice_mask_vocab = nice_mask

    def __call__(self, batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])          # (B,L)
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = input_ids.clone()

        B, L = input_ids.shape
        dev = input_ids.device

        # base masking probability
        prob = torch.full((B, L), self.random_mlm_prob, device=dev)

        # don't mask specials
        special = (input_ids == self.pad_id) | (input_ids == self.cls_id) | (input_ids == self.sep_id)
        prob.masked_fill_(special, 0.0)

        # first [SEP] position (because padding keeps [SEP] earlier than PAD)
        sep_pos = (input_ids == self.sep_id).float().argmax(dim=1)  # (B,)

        # last [APP] position (robust: if no [APP], set to 1)
        rev = torch.flip(input_ids, dims=[1])
        rev_app = (rev == self.app_id)
        has_app = rev_app.any(dim=1)
        rev_idx = rev_app.float().argmax(dim=1)
        last_app_pos = (L - 1) - rev_idx
        last_app_pos = torch.where(has_app, last_app_pos, torch.ones_like(last_app_pos))
        last_app_pos = torch.clamp(last_app_pos, min=1)

        # first [APP_END] after last_app_pos else fallback to sep
        end_pos = []
        for b in range(B):
            start = int(last_app_pos[b].item()) + 1
            end = int(sep_pos[b].item()) if int(sep_pos[b].item()) > 0 else L
            seg = input_ids[b, start:end]
            idx = (seg == self.app_end_id).nonzero(as_tuple=False)
            end_pos.append(start + int(idx[0].item()) if idx.numel() > 0 else end)
        end_pos = torch.tensor(end_pos, device=dev)

        positions = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
        in_last_event = (positions > last_app_pos.unsqueeze(1)) & (positions < end_pos.unsqueeze(1))

        item_flag = self.item_mask_vocab.to(dev)[input_ids]
        if self.mask_nice:
            nice_flag = self.nice_mask_vocab.to(dev)[input_ids]
            force_mask = in_last_event & (item_flag | nice_flag)
        else:
            force_mask = in_last_event & item_flag

        # force masking in last event
        prob[force_mask] = 1.0

        # sample final mask positions
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100

        # 80% -> [MASK]
        replace = (torch.bernoulli(torch.full((B, L), 0.8, device=dev)).bool()) & masked
        input_ids[replace] = self.mask_id

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ============================================================
# 2) EVALUATION (same idea as your code, just compact)
# ============================================================
@torch.no_grad()
def evaluate(model, loader, item_mask_vocab, topk_list=(1, 5, 10), max_batches=None):
    model.eval()
    total_loss = 0.0
    total_correct_all, total_masked_all = 0, 0
    total_item = 0
    correct_item_topk = {k: 0 for k in topk_list}

    for bi, batch in enumerate(loader, start=1):
        if max_batches is not None and bi > max_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()

        logits = out.logits
        labels = batch["labels"]
        mask = labels != -100

        if mask.any():
            preds = logits.argmax(dim=-1)
            total_correct_all += (preds[mask] == labels[mask]).sum().item()
            total_masked_all += mask.sum().item()

        # ITEM-only topk
        item_flag = item_mask_vocab.to(device)[labels.clamp(min=0)]
        item_pos = mask & item_flag
        if item_pos.any():
            true_item = labels[item_pos]
            idx = item_pos.nonzero(as_tuple=False)  # (N,2)
            gathered = logits[idx[:, 0], idx[:, 1], :]  # (N,V)

            total_item += true_item.numel()
            for k in topk_list:
                topk = torch.topk(gathered, k=k, dim=-1).indices
                correct_item_topk[k] += (topk == true_item.unsqueeze(1)).any(dim=1).sum().item()

    denom = max(1, bi)
    return {
        "loss": total_loss / denom,
        "acc_all": (total_correct_all / total_masked_all) if total_masked_all else 0.0,
        "masked_all": int(total_masked_all),
        "item_count": int(total_item),
        "item_topk": {k: (correct_item_topk[k] / total_item) if total_item else 0.0 for k in topk_list},
    }


# ============================================================
# 3) MODEL (Architecture) + OPTIMIZER + SCHEDULER
# ============================================================
def build_model(tokenizer: BertTokenizer):
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=HIDDEN_SIZE * 4,
        max_position_embeddings=MAX_LEN + 2,
        type_vocab_size=1,
    )
    return BertForMaskedLM(config)


def build_optim_sched(model, train_loader_len: int):
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = train_loader_len * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


# ============================================================
# 4) TRAINING LOOP (4 steps: forward -> loss -> backward -> update)
# ============================================================
def train_one_epoch(model, train_loader, optimizer, scheduler):
    model.train()
    total = 0.0
    pbar = tqdm(train_loader, desc="Train", leave=False)

    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        # (1) Forward pass
        outputs = model(**batch)

        # (2) Loss
        loss = outputs.loss

        # (3) Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # (4) Update
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total / max(1, len(train_loader))


def main():
    seed_all(SEED)
    print("Device:", device)
    print("Loading:", PICKLE_PATH)

    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)

    tokenizer = BertTokenizer(
        vocab_file=data["vocab_file"],
        do_lower_case=False,
        do_basic_tokenize=False
    )

    train_seqs = data["train"]
    valid_seqs = data["valid"]
    print(f"Train: {len(train_seqs):,} | Valid: {len(valid_seqs):,}")

    # ---- Step 1: DataLoader ----
    train_ds = SeqIdsDataset(train_seqs, MAX_LEN, tokenizer.pad_token_id, tokenizer.sep_token_id)
    valid_ds = SeqIdsDataset(valid_seqs, MAX_LEN, tokenizer.pad_token_id, tokenizer.sep_token_id) if len(valid_seqs) else None

    train_collator = ForecastingCollator(tokenizer, TRAIN_RANDOM_MLM_PROB, MASK_NICE_IN_LAST_EVENT)
    eval_collator  = ForecastingCollator(tokenizer, EVAL_RANDOM_MLM_PROB,  MASK_NICE_IN_LAST_EVENT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=train_collator)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=eval_collator) if valid_ds else None

    # ---- Step 2: Model (Architecture) ----
    model = build_model(tokenizer).to(device)

    # ---- Step 3: Optimizer + Scheduler ----
    optimizer, scheduler = build_optim_sched(model, len(train_loader))

    best_dir   = os.path.join(OUTPUT_DIR, "best_model")
    latest_dir = os.path.join(OUTPUT_DIR, "latest_model")
    tok_dir    = os.path.join(OUTPUT_DIR, "tokenizer")
    best_val = float("inf")

    print("\nModel structure:")
    print(f"  vocab={tokenizer.vocab_size}, hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, heads={NUM_HEADS}, max_len={MAX_LEN}")
    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}/{EPOCHS} | TrainLoss: {train_loss:.4f}")

        if valid_loader is not None:
            val = evaluate(model, valid_loader, train_collator.item_mask_vocab, topk_list=TOPK_LIST)
            tops = " ".join([f"Item@{k}:{val['item_topk'][k]:.4f}" for k in TOPK_LIST])
            print(f"           ValidLoss: {val['loss']:.4f} | AccAll:{val['acc_all']:.4f} | {tops}")

            if val["loss"] < best_val:
                best_val = val["loss"]
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(tok_dir)
                print("           >> Saved BEST")

        # save latest every epoch
        model.save_pretrained(latest_dir)
        tokenizer.save_pretrained(tok_dir)

    print("\nDone.")
    print("Best:  ", best_dir)
    print("Latest:", latest_dir)
    print("Tok:   ", tok_dir)


if __name__ == "__main__":
    main()
