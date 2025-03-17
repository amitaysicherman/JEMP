import torch
from torch.utils.data import Dataset
from transformers import T5Config, T5ForConditionalGeneration, Trainer, TrainingArguments
import numpy as np
from transformers import AutoTokenizer
import random
import os
from utils import load_state_dict_from_last_cp

class TranslationDataset(Dataset):
    def __init__(self, split, is_retro=False, parouts_context=1, max_len=5, max_mol_len=75, base_dir="USPTO"):
        self.max_len = 10
        self.max_mol_len = 75
        self.max_len = max_len
        self.max_mol_len = max_mol_len
        with open(f"data/{base_dir}/{split}-src.txt") as f:
            self.src = f.read().splitlines()
        with open(f"data/{base_dir}/{split}-tgt.txt") as f:
            self.tgt = f.read().splitlines()
        # with open(f"data/{base_dir}/{split}-cont.txt") as f:
        #     self.cont = f.read().splitlines()
        if is_retro:
            self.src, self.tgt = self.tgt, self.src

        # if (base_dir == "USPTO" and retro == 0) or (base_dir == "PaRoutes" and retro == 1 and parouts_context == 1):
        #     self.src = [s + "." + c for s, c in zip(self.src, self.cont)]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {"input_ids": self.src[idx], "labels": self.tgt[idx]}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[0]
    predictions = np.argmax(predictions, axis=-1)
    is_pad = labels == -100
    correct_or_pad = (predictions == labels) | is_pad
    perfect_match_accuracy = correct_or_pad.all(axis=1).mean()
    correct_not_pad = (predictions == labels) & ~is_pad
    token_accuracy = correct_not_pad.sum() / (~is_pad).sum()

    return {
        "sample_accuracy": perfect_match_accuracy,
        "token_accuracy": token_accuracy
    }
def collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_tokens = tokenizer(input_ids, padding="longest", return_tensors="pt")
    label_tokens = tokenizer(labels, padding="longest", return_tensors="pt")
    return {
        "input_ids": input_tokens.input_ids,
        "attention_mask": input_tokens.attention_mask,
        "labels": label_tokens.input_ids
    }
def main(retro=False, batch_size=256, parouts=0):
    # Build vocabulary and create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=512,  # Hidden size
        d_ff=2048,  # Intermediate feed-forward size
        num_layers=6,  # Number of encoder/decoder layers
        num_heads=4,  # Number of attention heads
        is_encoder_decoder=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )

    # Initialize model from configuration
    model = T5ForConditionalGeneration(config)
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Prepare datasets
    train_dataset = TranslationDataset(
        "train",
        is_retro=retro,
        parouts_context=parouts
    )
    val_dataset = TranslationDataset(
        "valid",
        is_retro=retro,
        parouts_context=parouts
    )
    train_subset_random_indices = random.sample(range(len(train_dataset)), len(val_dataset))
    train_subset = torch.utils.data.Subset(train_dataset, train_subset_random_indices)

    # Training arguments
    output_suf = "retro" if retro else "forward"
    res_dir = f"res/{output_suf}"
    os.makedirs(res_dir, exist_ok=True)
    state_dict = load_state_dict_from_last_cp(res_dir)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded model from last checkpoint")


    training_args = TrainingArguments(
        output_dir=res_dir,
        evaluation_strategy="steps",
        learning_rate=2e-4,  # Higher learning rate since we're training from scratch
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=100,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=100,  # More epochs since we're training from scratch
        logging_steps=500,
        save_steps=2500,
        warmup_steps=2000,
        logging_dir=f"./logs/{output_suf}",
        lr_scheduler_type="constant",
        load_best_model_at_end=True,
        eval_steps=2500,
        metric_for_best_model="eval_validation_token_accuracy",
        save_only_model=True,
        # auto_find_batch_size=True,
        save_safetensors=False,


    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={'validation': val_dataset, "train": train_subset},
        compute_metrics=compute_metrics,
        data_collator=lambda batch: collate_fn(batch, tokenizer)

    )
    score = trainer.evaluate()
    print(score)

    # Train the model
    trainer.train(resume_from_checkpoint=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--retro", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--parouts", type=int, default=0)
    args = parser.parse_args()
    retro = args.retro
    main(retro, args.batch_size, args.parouts)
