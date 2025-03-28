import argparse
from typing import Optional, Tuple, Union

from transformers import (
    T5ForConditionalGeneration,
    T5Config,
    TrainingArguments,
    Trainer,
    AutoTokenizer
)
from torch.utils.data import Dataset as TorchDataset
from os.path import join as pjoin
import numpy as np
import torch
import os
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import Seq2SeqLMOutput

from JERP.trie import Trie, build_trie, build_mask_from_trie


def load_file(file_path):
    """Load text file"""
    with open(file_path) as f:
        texts = f.read().splitlines()
    return texts


def load_files(base_dir="data/Reactzyme/data"):
    """Load source and target text files"""
    src_train = load_file(pjoin(base_dir, "train_reaction.txt"))
    tgt_train = load_file(pjoin(base_dir, "train_enzyme.txt"))
    src_test = load_file(pjoin(base_dir, "test_reaction.txt"))
    tgt_test = load_file(pjoin(base_dir, "test_enzyme.txt"))
    return src_train, tgt_train, src_test, tgt_test


class SrcTgtDataset(TorchDataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length=512):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        model_inputs = self.tokenizer(
            src_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )

        labels = self.tokenizer(
            tgt_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
        return model_inputs


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = predictions[0].argmax(-1)

    # Create mask for non-padding tokens (-100 is the default ignore index)
    non_pad_mask = labels != -100

    # Token-level accuracy (only on non-padded tokens)
    token_correct = 0
    token_total = 0

    # Sample-level accuracy (exact match)
    sample_correct = 0
    sample_total = len(labels)

    for i in range(len(labels)):
        # Get mask for this sequence
        seq_mask = non_pad_mask[i]

        # Extract non-padded tokens for this sequence
        seq_true = labels[i][seq_mask]
        seq_pred = predictions[i][seq_mask]

        # Count correct tokens
        token_correct += np.sum(seq_pred == seq_true)
        token_total += len(seq_true)

        # Check if entire sequence is correct (exact match)
        if np.array_equal(seq_pred, seq_true):
            sample_correct += 1

    # Calculate accuracies
    token_accuracy = token_correct / token_total if token_total > 0 else 0
    sample_accuracy = sample_correct / sample_total if sample_total > 0 else 0

    return {
        "token_accuracy": token_accuracy,
        "sample_accuracy": sample_accuracy
    }


def get_last_checkpoint(output_dir):
    all_cp_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if len(all_cp_dirs) == 0:
        return None
    all_cp_dirs.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
    last_cp_dir = all_cp_dirs[0]
    last_cp_path = os.path.join(output_dir, last_cp_dir)
    return last_cp_path


class ModelWithTrie(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, trie: Trie):
        super().__init__(config)
        self.trie = trie

    def forward(
            self, input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            **kwargs
    ):
        # apply the super class forward method
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        decoder_input = torch.cat([labels.new_ones((labels.shape[0], 1)) * self.config.decoder_start_token_id, labels[:, :-1]],
                           dim=1)
        trie_mask = build_mask_from_trie(self.trie, decoder_input, self.config.vocab_size)
        # replace   zero with -inf
        trie_mask = trie_mask.masked_fill(trie_mask == 0, -1e6)
        trie_mask = trie_mask.masked_fill(trie_mask == 1, 0)
        trie_mask = trie_mask.to(outputs.logits.device)

        outputs.logits = outputs.logits + trie_mask
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        labels = labels.to(outputs.logits.device)
        outputs.loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

        return outputs


def main(args):
    print("Loading text files...")
    src_train, tgt_train, src_test, tgt_test = load_files()

    if args.debug:
        src_train, tgt_train, src_test, tgt_test = src_train[:2], tgt_train[:2], src_test[:2], tgt_test[:2]
    assert len(src_train) == len(tgt_train)
    assert len(src_test) == len(tgt_test)
    # Create output directory if it doesn't exist

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D", trust_remote_code=True)
    tgt_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    n_tokens = len(tokenizer.get_vocab())
    for key in tgt_tokenizer.vocab:
        if key not in tokenizer.get_vocab():
            tokenizer.add_tokens(key)
    print(f"Number of tokens: {n_tokens} -> {len(tokenizer.get_vocab())}")
    # Prepare dataset
    print("Preparing dataset...")

    train_dataset = SrcTgtDataset(src_train, tgt_train, tokenizer, max_length=args.max_length)
    test_dataset = SrcTgtDataset(src_test, tgt_test, tokenizer, max_length=args.max_length)

    train_small_indices = np.random.choice(len(train_dataset), len(test_dataset), replace=False)
    train_small_dataset = torch.utils.data.Subset(train_dataset, train_small_indices)

    if args.size == "s":
        d_model = 256
        d_ff = 1024
        num_heads = 4
        num_layers = 4
    elif args.size == "m":
        d_model = 512
        d_ff = 2048
        num_heads = 8
        num_layers = 6
    elif args.size == "l":
        d_model = 1024
        d_ff = 4096
        num_heads = 16
        num_layers = 12
    else:
        raise ValueError(f"Unknown model size: {args.size}")

    config = T5Config(
        vocab_size=len(tokenizer.get_vocab()),
        n_positions=args.max_length,
        decoder_start_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        is_encoder_decoder=True,
        bos_token_id=tokenizer.bos_token_id,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,

    )
    if args.debug:
        config.d_model = 64
        config.d_ff = 256
        config.num_heads = 2
        config.num_layers = 2
    if args.trie:
        trie = build_trie(tgt_train + tgt_test, tokenizer, tokenizer.pad_token_id)
        model = ModelWithTrie(config, trie)
    else:
        model = T5ForConditionalGeneration(config)
    print("Model created!")
    print(model)
    print(f"Model parameters:{model.num_parameters():,}")
    output_dir = f"trans/model/{args.learning_rate}"
    log_dir = f"trans/logs/{args.learning_rate}"
    if args.trie:
        output_dir += "_trie"
        log_dir += "_trie"
    print(args.batch_size, len(train_dataset), len(test_dataset))
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_accumulation_steps=30,
        # weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        logging_dir=log_dir,
        logging_steps=args.log_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        lr_scheduler_type="constant",
        load_best_model_at_end=True,
        metric_for_best_model="eval_test_token_accuracy",
        report_to=[args.report_to],
        # label_names=["labels"],
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"test": test_dataset, "train": train_small_dataset},
        compute_metrics=compute_metrics
    )
    # Train model
    print("Training model...")
    trainer.train(resume_from_checkpoint=get_last_checkpoint(output_dir))

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T5 model for translation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--log_steps", type=int, default=500, help="Log training steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluate every n steps")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save model every n steps")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report to tensorboard or wandb")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--size", type=str, default="m", help="Model size: s, m, l")
    parser.add_argument("--trie", action="store_true", help="Use trie for training")
    args = parser.parse_args()
    if args.size == "l":
        args.batch_size = 16
    else:
        args.batch_size = 64
    if args.debug:
        args.size = "s"
        args.batch_size = 2
        args.epochs = 10000
        args.log_steps = 100
        args.eval_steps = 100
        args.save_steps = 50000000
        args.report_to = "none"
        args.learning_rate = 0.001

    main(args)
