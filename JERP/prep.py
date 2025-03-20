import torch
import os
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

# Load the pretrained tokenizers
enzyme_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D", trust_remote_code=True)
reaction_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

max_enzyme_len = 1024
max_reaction_len = 1024


def remove_stereo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


skip_count = 0
total_count = 0
output_base = "data/Reactzyme/data/"
reaction_length = []
enzyme_length = []
os.makedirs(output_base, exist_ok=True)
for split in ["train_val", "test"]:
    input_file = f"data/Reactzyme/enzyme_smi_split/positive_{split}_seq_smi.pt"
    output_reaction_file = f"{output_base}/{split.split('_')[0]}_reaction.txt"
    output_enzyme_file = f"{output_base}/{split.split('_')[0]}_enzyme.txt"
    data = torch.load(input_file)

    with open(output_reaction_file, "w") as reaction_file:
        with open(output_enzyme_file, "w") as enzyme_file:
            pbar = tqdm(data.values(), desc=f"Processing {split}", total=len(data))
            for reaction, enzyme in pbar:
                total_count += 1
                reaction = remove_stereo(reaction)
                pbar.set_postfix({"skip_count": skip_count, "total_count": total_count,
                                  "% skipped": f"{skip_count / total_count:.2%}"})
                reaction_tokens = reaction_tokenizer(reaction, truncation=False, padding="do_not_pad")["input_ids"]
                enzyme_tokens = enzyme_tokenizer(enzyme, truncation=False, padding="do_not_pad")["input_ids"]
                if len(reaction_tokens) > max_reaction_len or len(enzyme_tokens) > max_enzyme_len:
                    skip_count += 1
                    continue
                reaction_file.write(reaction + "\n")
                enzyme_file.write(enzyme + "\n")
                reaction_length.append(len(reaction))
                enzyme_length.append(len(enzyme))
