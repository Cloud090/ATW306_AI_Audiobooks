"""
Train a LoRA adapter for Orpheus-3B on SNAC-encoded CREMA-D + Steven data.

This script:
  - Loads the base Orpheus-3B model via Unsloth.
  - Enables LoRA on key transformer modules.
  - Loads a SNAC-encoded DatasetDict from disk.
  - Packs text + acoustic codes into a single token sequence.
  - Trains a LoRA adapter using Transformers' Trainer.
  - Saves both the LoRA-only checkpoint and an optional merged fp16 model.

This is a training-only script used during development, not part of the final runtime pipeline.
"""
import unsloth  # MUST be imported first
import torch
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel


# ========================= Config =========================

# Base Orpheus-3B checkpoint (Unsloth version).
BASE = "unsloth/orpheus-3b-0.1-pretrained"

# Local path to SNAC-encoded dataset.
DATA = "data/cremad_plus_steven_3x_snac_24k"

# Special token IDs (keep in sync with the Orpheus/Unsloth notebook).
TOKENISER_LEN = 128256
START_OF_TEXT = 128000
END_OF_TEXT   = 128009
START_HUMAN   = TOKENISER_LEN + 3
END_HUMAN     = TOKENISER_LEN + 4
START_AI      = TOKENISER_LEN + 5
END_AI        = TOKENISER_LEN + 6
START_SPEECH  = TOKENISER_LEN + 1
END_SPEECH    = TOKENISER_LEN + 2
PAD_ID        = 128263

# Maximum total sequence length (text + acoustic codes).
MAX_SEQ_LEN = 2048


# ========================= Model + dataset load =========================

# 1) Load base model in 16-bit for quality (recommended for TTS LoRA).
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=2048,
    dtype=None, # bf16 under the hood via Trainer
    load_in_4bit=False,
)

# 2) Enable LoRA on key transformer modules.
model = FastLanguageModel.get_peft_model(
    model,
    r=64, lora_alpha=64, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)

# 3) Load SNAC-encoded dataset.
dsd = load_from_disk(DATA)
train_ds = dsd["train"]
val_ds   = dsd["validation"]


# ========================= Packing + collate =========================

def pack(example):
    """
    Pack one example into a single token sequence for training.

    Layout:
        [START_HUMAN]
        <spk=ID> text ... END_OF_TEXT
        [END_HUMAN]
        [START_AI] [START_SPEECH]
        SNAC acoustic codes ...
        [END_SPEECH] [END_AI]

    This matches the Orpheus training convention, where:
      - The "human" section is the conditioning text.
      - The "AI speech" section is the acoustic code sequence to predict.
    """
    speaker = example.get("speaker")
    if speaker:
        conditioned_text = f"<spk={speaker}> {example['text']}"
    else:
        conditioned_text = example["text"]

    # Encode text + speaker tags.
    text_ids = tokenizer.encode(conditioned_text, add_special_tokens=True)
    text_ids.append(END_OF_TEXT)

    # Build full sequence: human -> AI speech.
    ids = (
        [START_HUMAN] + text_ids + [END_HUMAN] +
        [START_AI, START_SPEECH] + example["codes_list"] + [END_SPEECH, END_AI]
    )

    # Truncate if it would exceed the models maximum sequence length. (Happens with a few longer synthetic examples)
    if len(ids) > MAX_SEQ_LEN:
        ids = ids[:MAX_SEQ_LEN]

    example["input_ids"] = ids
    example["labels"] = ids
    example["attention_mask"] = [1] * len(ids)
    return example


# Keep only the packed fields needed for training.
keep = ["input_ids","labels","attention_mask"]
train_ds = train_ds.map(pack, remove_columns=[c for c in train_ds.column_names if c not in keep])
val_ds   = val_ds.map(pack,   remove_columns=[c for c in val_ds.column_names   if c not in keep])

def collate(batch):
    """
    Custom data collator that:
      - Converts lists of token IDs into tensors.
      - Pads sequences in the batch to the same length with PAD_ID.

    This ensures input_ids, labels and attention_mask are all aligned and
    ready for the Trainer.
    """
    keys = ["input_ids","labels","attention_mask"]
    # Turn each sample's list into a tensor.
    out = {k: [torch.tensor(b[k]) for b in batch] for k in keys}
    # Pad across the batch.
    out = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True, padding_value=PAD_ID) for k,v in out.items()}
    return out


# ========================= Training setup =========================

# 5) Training args (Setup for a 12 GB RTX 4070).
args = TrainingArguments(
    output_dir="checkpoints/orpheus_lora_cremad_plus_steven_mspk_v5",   # checkpoint dir
    per_device_train_batch_size=1,  # small batch size for VRAM limits
    gradient_accumulation_steps=4,   # effective batch size = 4
    learning_rate=2e-4,               # LoRA-specific LR
    warmup_steps=200,                 # warmup for stability
    num_train_epochs=2,              # start with 2 epochs (~13k steps effective)
    logging_steps=50,                 # log every 50 steps
    eval_strategy="no",                # no eval during training
    #eval_steps=200,                  # eval every 200 steps
    save_steps=1000,                 # save every 1000 steps
    save_total_limit=3,               # keep last 3 checkpoints
    weight_decay=1e-3,                # regularization
    lr_scheduler_type="linear",        # linear LR decay
    optim="adamw_8bit",              # VRAM-friendly optimizer
    bf16=True,                       # 4070 supports bf16 well
    report_to="none",                  # no logging integration
    seed=2025,                      # for reproducibility
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate,
)


# ========================= Train + save =========================

# 6) Run training.
trainer.train()

# Save LoRA adapters to a final checkpoint folder.
lora_out_dir = "checkpoints/orpheus_lora_cremad_plus_steven_mspk_v5/final"  # local destination
trainer.save_model(lora_out_dir)
tokenizer.save_pretrained(lora_out_dir)
print(f"Saved LoRA to {lora_out_dir}")

# 7) Merge LoRA into a full fp16 model for simpler inference later.
try:
    merged = model.merge_and_unload()   # Unsloth helper to apply LoRA into the base.
    merged.save_pretrained("checkpoints/orpheus_merged_cremad_plus_steven_fp16_mspk_v5")
    print("Saved merged model to checkpoints/orpheus_merged_cremad_plus_steven_fp16_mspk_v5")

# In some environments (or with some PEFT configs) merge may not be supported.
except Exception as e:
    print("[info] merge skipped/not supported in this environment:", e)

"""
No args this time, just run with:
python train.py
"""