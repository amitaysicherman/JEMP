import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
import os
import random
from transformers import AutoTokenizer
from transformers import BertGenerationDecoder, BertGenerationConfig, BertGenerationEncoder
from decoder.main import create_model
from transformers import AutoModel
import numpy as np

DEBUG = False
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
num_layers = {'s': 2, 'sm': 4, 'm': 6, 'l': 24}
num_heads = {'s': 2, 'sm': 4, 'm': 4, 'l': 8}
DROPOUT = 0.1


def size_to_configs(size, hidden_size, tokenizer, dropout=DROPOUT):
    size_args = dict(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers[size],
        num_attention_heads=num_heads[size],
        intermediate_size=hidden_size * 4,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )
    common_args = dict(
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        is_encoder_decoder=True,
    )
    encoder_config = BertGenerationConfig(**common_args, **size_args, is_decoder=False)
    decoder_config = BertGenerationConfig(**common_args, **size_args, is_decoder=True, add_cross_attention=True)
    return encoder_config, decoder_config


class ReactionMolsDataset(Dataset):
    def __init__(self, base_dir="USPTO", split="train", max_mol_len=75, max_len=5, parouts_context=True,
                 is_retro=False):
        self.max_len = max_len
        self.max_mol_len = max_mol_len
        with open(f"data/{base_dir}/{split}-src.txt") as f:
            self.src = f.read().splitlines()
        with open(f"data/{base_dir}/{split}-tgt.txt") as f:
            self.tgt = f.read().splitlines()
        with open(f"data/{base_dir}/{split}-cont.txt") as f:
            self.cont = f.read().splitlines()

        self.is_retro = is_retro
        self.is_uspto = base_dir == "USPTO"
        self.parouts_context = parouts_context

        if DEBUG:
            self.src = self.src[:10]
            self.tgt = self.tgt[:10]
            self.cont = self.cont[:10]

        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

        self.empty = {"input_ids": torch.tensor([self.tokenizer.pad_token_id] * self.max_mol_len),
                      "attention_mask": torch.tensor([0] * self.max_mol_len)}

    def preprocess_line(self, line):
        if line == "":
            return {
                'input_ids': torch.zeros(self.max_len, self.max_mol_len,
                                         dtype=torch.long) + self.tokenizer.pad_token_id,
                'token_attention_mask': torch.zeros(self.max_len, self.max_mol_len, dtype=torch.long),
                'mol_attention_mask': torch.zeros(self.max_len, dtype=torch.long)
            }
        mols = line.strip().split(".")
        tokens = [self.tokenizer(m, padding="max_length", truncation=True, max_length=self.max_mol_len,
                                 return_tensors="pt") for m in mols]
        tokens = [{k: v.squeeze(0) for k, v in t.items()} for t in tokens]
        num_mols = len(tokens)
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        attention_mask[:num_mols] = 1
        while len(tokens) < self.max_len:
            tokens.append({k: v.clone() for k, v in self.empty.items()})
        input_ids = torch.stack([t['input_ids'].squeeze(0) for t in tokens])
        token_attention_mask = torch.stack([t['attention_mask'].squeeze(0) for t in tokens])
        return {
            'input_ids': input_ids,  # Shape: (max_seq_len, 75)
            'token_attention_mask': token_attention_mask,  # Shape: (max_seq_len, 75)
            'mol_attention_mask': attention_mask,  # Shape: (max_seq_len,)
        }

    def __getitem__(self, idx):

        src, tgt, cont = self.src[idx], self.tgt[idx], self.cont[idx]
        src_tokens = self.preprocess_line(src)
        tgt_tokens = self.preprocess_line(tgt)
        cont_tokens = self.preprocess_line(cont)
        if self.is_uspto:

            if self.is_retro:
                src_tokens, tgt_tokens = tgt_tokens, src_tokens
            else:
                src_tokens = {
                    'input_ids': torch.cat((src_tokens['input_ids'], cont_tokens['input_ids']), dim=0),
                    'token_attention_mask': torch.cat(
                        (src_tokens['token_attention_mask'], cont_tokens['token_attention_mask']),
                        dim=0),
                    'mol_attention_mask': torch.cat(
                        (src_tokens['mol_attention_mask'], cont_tokens['mol_attention_mask']),
                        dim=0)
                }
        else:
            assert self.is_retro
            src_tokens, tgt_tokens = tgt_tokens, src_tokens
            if self.parouts_context:
                src_tokens = {
                    'input_ids': torch.cat((src_tokens['input_ids'], cont_tokens['input_ids']), dim=0),
                    'token_attention_mask': torch.cat(
                        (src_tokens['token_attention_mask'], cont_tokens['token_attention_mask']),
                        dim=0),
                    'mol_attention_mask': torch.cat(
                        (src_tokens['mol_attention_mask'], cont_tokens['mol_attention_mask']),
                        dim=0)
                }

        return {
            "src_input_ids": src_tokens['input_ids'],
            "src_token_attention_mask": src_tokens['token_attention_mask'],
            "src_mol_attention_mask": src_tokens['mol_attention_mask'],
            "tgt_input_ids": tgt_tokens['input_ids'],
            "tgt_token_attention_mask": tgt_tokens['token_attention_mask'],
            "tgt_mol_attention_mask": tgt_tokens['mol_attention_mask'],
        }

    def __len__(self):
        return len(self.src)


class MVM(nn.Module):
    def __init__(self, config_enc, config_dec, encoder=None, decoder=None, is_trainable_encoder=False):
        super().__init__()
        self.config_enc = config_enc
        self.config_dec = config_dec
        self.encoder = encoder
        self.is_trainable_encoder = is_trainable_encoder
        self.decoder = decoder
        self.bert_encoder = BertGenerationEncoder(config_enc)
        self.bert_decoder = BertGenerationDecoder(config_dec)
        self.decoder_start_token = torch.nn.Parameter(torch.randn(1, config_dec.hidden_size))

    def get_mol_embeddings(self, input_ids, token_attention_mask):
        batch_size, max_seq_len, seq_len = input_ids.shape
        with torch.set_grad_enabled(self.is_trainable_encoder):
            embeddings = self.encoder(input_ids=input_ids.view(-1, seq_len),
                                      attention_mask=token_attention_mask.view(-1, seq_len)).pooler_output
        return embeddings.view(batch_size, max_seq_len, -1)

    def forward(self, src_input_ids, src_token_attention_mask, src_mol_attention_mask, tgt_input_ids,
                tgt_token_attention_mask, tgt_mol_attention_mask):
        src_embeddings = self.get_mol_embeddings(src_input_ids, src_token_attention_mask)
        src_seq_mask = src_mol_attention_mask.float()

        bert_encoder_output = self.bert_encoder(inputs_embeds=src_embeddings, attention_mask=src_seq_mask)[
            'last_hidden_state']

        tgt_embeddings = self.get_mol_embeddings(tgt_input_ids, tgt_token_attention_mask)
        decoder_start_token = self.decoder_start_token.unsqueeze(1).expand(tgt_embeddings.size(0), 1, -1)
        tgt_embeddings = torch.cat([decoder_start_token, tgt_embeddings], dim=1)

        bert_decoder_output = self.bert_decoder(inputs_embeds=tgt_embeddings[:, :-1],
                                                attention_mask=tgt_mol_attention_mask,
                                                encoder_hidden_states=bert_encoder_output,
                                                encoder_attention_mask=src_seq_mask,
                                                output_hidden_states=True)['hidden_states'][-1]

        bs, seq_len, _ = bert_decoder_output.size()
        labels = tgt_input_ids.view(bs * seq_len, -1).clone()
        labels[tgt_token_attention_mask.view(bs * seq_len, -1) == 0] = -100
        output= self.decoder(
            input_ids=tgt_input_ids.view(bs * seq_len, -1),
            attention_mask=tgt_token_attention_mask.view(bs * seq_len, -1),
            encoder_outputs=bert_decoder_output.view(bs * seq_len, -1),
            labels=labels
        )
        output.logits = output.logits.argmax(dim=-1).view(bs, seq_len, -1)

        return output


def compute_metrics(eval_pred):
    predictions, (labels, tgt_token_attention_mask, _) = eval_pred
    labels_mask = tgt_token_attention_mask.astype(bool)
    correct_predictions = (predictions == labels) & labels_mask
    token_accuracy = correct_predictions.sum() / labels_mask.sum()
    sample_mask = labels_mask.sum(axis=2) > 0  # Identify valid samples
    sample_correct = np.all(correct_predictions | ~labels_mask, axis=2)
    sample_accuracy = sample_correct[sample_mask].mean()

    row_correct = np.all(sample_correct | ~sample_mask, axis=1)
    row_accuracy = row_correct.mean()

    return {
        "token_accuracy": token_accuracy,
        "sample_accuracy": sample_accuracy,
        "row_accuracy": row_accuracy,
    }


def main(batch_size=32, num_epochs=10, lr=1e-4, size="m", train_encoder=False,
         train_decoder=False, cp=None, dropout=DROPOUT, parouts=False, parouts_context=True,
         eval_accumulation_steps=300, retro=True):
    base_dir = "USPTO" if not parouts else "PaRoutes"
    train_dataset = ReactionMolsDataset(split="train", parouts_context=parouts_context, base_dir=base_dir, is_retro=retro)
    val_dataset = ReactionMolsDataset(split="valid", parouts_context=parouts_context, base_dir=base_dir, is_retro=retro)
    if DEBUG:
        val_dataset = ReactionMolsDataset(split="train", parouts_context=parouts_context, base_dir=base_dir, is_retro=retro)
    train_subset_random_indices = random.sample(range(len(train_dataset)), len(val_dataset))
    train_subset = torch.utils.data.Subset(train_dataset, train_subset_random_indices)
    encoder_config, decoder_config = size_to_configs(size, 768, train_dataset.tokenizer, dropout=dropout)
    # Load encoder and decoder

    encoder = AutoModel.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct",
        deterministic_eval=True,
        trust_remote_code=True,
        use_safetensors=False  # Force using PyTorch format instead of safetensors
    )

    decoder, _ = create_model()
    state_dict = torch.load("decoder/results/checkpoint-195000/pytorch_model.bin", map_location=torch.device('cpu'))
    decoder.load_state_dict(state_dict, strict=True)

    encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad = train_encoder

    decoder.to(device)
    for param in decoder.parameters():
        param.requires_grad = train_decoder

    model = MVM(config_enc=encoder_config, config_dec=decoder_config, encoder=encoder, decoder=decoder,
                is_trainable_encoder=train_encoder)

    # Initialize MVM model with encoder and decoder
    if cp is not None:
        model.load_state_dict(torch.load(cp, map_location=device), strict=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    print(f"MODEL: {model.__class__.__name__}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

    # Update output suffix to include encoder/decoder training info
    output_suf = f"{size}_{lr}"
    if train_encoder:
        output_suf += "_train_enc"
    if train_decoder:
        output_suf += "_train_dec"
    if cp is not None:
        output_suf += "_cp"
    if dropout != DROPOUT:
        output_suf += f"_dropout-{dropout}"
    if parouts:
        output_suf += "_parouts"
        if parouts_context:
            output_suf += "_context"
    if retro:
        output_suf += "_retro"

    os.makedirs(f"results/{output_suf}", exist_ok=True)
    train_args = TrainingArguments(
        output_dir=f"results/{output_suf}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=eval_accumulation_steps,
        logging_dir=f"logs/{output_suf}",
        logging_steps=500,
        save_steps=2500,
        evaluation_strategy="steps",
        eval_steps=2500 if not DEBUG else 50,
        save_total_limit=1,
        load_best_model_at_end=True,
        save_safetensors=False,
        gradient_accumulation_steps=1,
        metric_for_best_model="eval_validation_token_accuracy",
        report_to="tensorboard",
        learning_rate=lr,
        lr_scheduler_type='constant',
        label_names=['tgt_input_ids', 'tgt_token_attention_mask', 'tgt_mol_attention_mask'],
        seed=42,
        save_only_model=True,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset={'validation': val_dataset, "train": train_subset},
        compute_metrics=lambda x: compute_metrics(x),
    )
    print(trainer.evaluate())
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--eval_accumulation_steps", type=float, default=300)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--size", type=str, default="m")
    parser.add_argument("--train_encoder", type=int, default=1)
    parser.add_argument("--train_decoder", type=int, default=1)
    parser.add_argument("--cp", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--parouts", type=int, default=0)
    parser.add_argument("--parouts_context", type=int, default=1)
    parser.add_argument("--retro", type=int, default=1)

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    main(
        args.batch_size,
        args.num_epochs,
        args.lr,
        args.size,
        bool(args.train_encoder),
        bool(args.train_decoder),
        args.cp,
        args.dropout,
        bool(args.parouts),
        bool(args.parouts_context),
        args.eval_accumulation_steps,
        bool(args.retro)
    )
