import numpy as np
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments
import torch.nn as nn
from transformers import AutoModel, T5Config, AutoTokenizer, PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack
from os.path import join as pjoin
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.mps.is_available():
#     device = torch.device("mps")


def _shift_right(input_ids, decoder_start_token_id, pad_token_id):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class FASTADataset(Dataset):
    def __init__(self, split="train"):
        base_dir = "data/Reactzyme/data"
        data_file = pjoin(base_dir, f"{split}_enzyme.txt")
        self.fasta = []
        with open(data_file) as f:
            for line in f:
                self.fasta.append(line.strip())
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D", trust_remote_code=True)
        self.esm = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D", trust_remote_code=True)
        self.esm.eval().to(device)
        for param in self.esm.parameters():
            param.requires_grad = False

    def __len__(self):
        return len(self.fasta)

    def __getitem__(self, idx):
        smile = self.fasta
        tokens = self.tokenizer(smile, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        with torch.no_grad():
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = self.esm(**tokens)
        tokens["encoder_hidden_states"] = outputs.last_hidden_state.detach().cpu()  # .mean(axis=-1)

        print(tokens["encoder_hidden_states"].shape)
        labels = tokens["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        tokens["labels"] = labels

        return tokens


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    mask = labels != -100
    total_tokens = mask.sum()
    correct_tokens = ((predictions == labels) & mask).sum()
    token_accuracy = correct_tokens / total_tokens
    correct_or_pad = (predictions == labels) | (~mask)
    correct_samples = correct_or_pad.all(axis=-1).sum()
    total_samples = len(labels)
    sample_accuracy = correct_samples / total_samples
    return {
        "token_accuracy": token_accuracy,
        "sample_accuracy": sample_accuracy,
    }


class ProteinEmbDecoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = T5Stack(config, embed_tokens)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, labels, encoder_outputs):
        decoder_input_ids = _shift_right(input_ids, self.config.decoder_start_token_id, self.config.pad_token_id)
        decoder_output = self.decoder(encoder_hidden_states=encoder_outputs, input_ids=decoder_input_ids)
        lm_logits = self.lm_head(decoder_output.last_hidden_state)
        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), ignore_index=-100)
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
        )


def create_model(debug=False):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D", trust_remote_code=True)
    if debug:
        config = T5Config(
            vocab_size=len(tokenizer),
            d_model=256,
            d_ff=512,
            num_layers=4,
            is_encoder_decoder=False,
            is_decoder=True,
            num_heads=4,
            decoder_start_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        config = T5Config(
            vocab_size=len(tokenizer),
            d_model=768,
            d_ff=2048,
            is_encoder_decoder=False,
            is_decoder=True,
            num_layers=6,
            num_decoder_layers=6,
            num_heads=8,
            decoder_start_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,


        )
    if debug:
        print(config)
    model = ProteinEmbDecoder(config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    print(
        f"Total parameters: {total_params:,},(trainable: {trainable_params:,}, non-trainable: {non_trainable_params:,})")
    return model, tokenizer


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model, tokenizer = create_model(args.debug)
    train_dataset = FASTADataset(split="train")
    eval_dataset = FASTADataset(split="test")
    DEBUG = args.debug
    if DEBUG:
        train_dataset = random_split(train_dataset, [2, len(train_dataset) - 2])[0]
        eval_dataset = random_split(eval_dataset, [2, len(eval_dataset) - 2])[0]

    output_dir = f"./esm_decoder/results"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10 if not DEBUG else 10000,
        per_device_train_batch_size=16 if not DEBUG else 2,
        per_device_eval_batch_size=16 if not DEBUG else 2,
        learning_rate=1e-4 if not DEBUG else 1e-3,
        logging_steps=1_000 if not DEBUG else 10,
        save_steps=5_000 if not DEBUG else 50000000,
        eval_accumulation_steps=2,
        eval_steps=5_000 if not DEBUG else 10,
        evaluation_strategy="steps",
        report_to=["tensorboard"] if not DEBUG else [],
        lr_scheduler_type="linear",
        warmup_steps=5_000 if not DEBUG else 500,
        load_best_model_at_end=True,
        metric_for_best_model="token_accuracy",
        save_safetensors=False,
        label_names=["labels"],
    )

    # Initialize trainer with evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda x: compute_metrics,
    )
    scores = trainer.evaluate()
    print(scores)

    trainer.train()
