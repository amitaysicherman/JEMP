from rdkit import Chem
from transformers import AutoTokenizer
from rdkit import RDLogger
from tqdm import tqdm
import pickle
import pandas as pd

RDLogger.DisableLog('rdApp.*')


class SmilesPreprocessor:
    def __init__(self, max_len=75, max_mols=5):
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.max_len = max_len
        self.max_mols = max_mols

    def format_mol(self, mol: Chem.Mol):
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

    def format_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return self.format_mol(mol)

    def smiles_to_tokens(self, smiles):
        return self.tokenizer(smiles, padding=False, truncation=False)['input_ids']

    def tokens_have_unk(self, tokens):
        return self.tokenizer.unk_token_id in tokens

    def count_tokens(self, tokens):
        return len(tokens)

    def format_filter(self, smiles):
        smiles = self.format_smiles(smiles)
        if smiles == "":
            return ""
        tokens = self.smiles_to_tokens(smiles)
        if self.tokens_have_unk(tokens):
            return ""
        if self.count_tokens(tokens) > self.max_len:
            return ""
        return smiles

    def multismilts_format_filter(self, milti_smiles, is_tgt=False):
        smiles = [self.format_filter(s) for s in milti_smiles.split(".")]
        if is_tgt and len(smiles) > 1:
            return ""
        if len(smiles) > self.max_mols:
            return ""
        if "" in smiles:
            return ""
        return ".".join(smiles)

    def prep_pubchem(self, input_file="data/pubchem/CID-SMILES", output_file="data/pubchem/smiles.txt"):
        with open(input_file, 'r', encoding='utf-8') as f:
            all_data = f.read().splitlines()
        pbar = tqdm(all_data)

        with open(output_file, 'w') as f:
            filter_count = 0
            total_count = 0
            for data in pbar:
                _, smiles = data.strip().split()
                smiles = self.format_filter(smiles)
                if smiles != "":
                    f.write(smiles + "\n")
                    filter_count += 1
                total_count += 1
                pbar.set_description(f"Filtered: {filter_count:,}/{total_count:,}[{filter_count / total_count:.2%}]")

    def prep_uspto(self, base_dir="data/USPTO", file_name="data.pickle"):
        input_file = f"{base_dir}/{file_name}"
        with open(input_file, 'rb') as f:
            data: pd.DataFrame = pickle.load(f)
        for split in ['train', 'valid', 'test']:
            data_split = data[data['set'] == split]
            with open(f"{base_dir}/{split}-src.txt", 'w') as f_src:
                with open(f"{base_dir}/{split}-tgt.txt", 'w') as f_tgt:
                    with open(f"{base_dir}/{split}-cont.txt", 'w') as f_con:
                        all_count = 0
                        filter_count = 0
                        pbar = tqdm(data_split.iterrows(), total=len(data_split))
                        for _, row in pbar:
                            s, t, c = row['reactants_mol'], row['products_mol'], row['reagents_mol']
                            s = self.multismilts_format_filter(self.format_mol(s))
                            t = self.multismilts_format_filter(self.format_mol(t), is_tgt=True)
                            if c is not None:
                                c = self.multismilts_format_filter(self.format_mol(c))
                                if c == "":
                                    continue
                            else:
                                c = ""
                            if s != "" and t != "":
                                f_src.write(s + "\n")
                                f_tgt.write(t + "\n")
                                f_con.write(c + "\n")
                                filter_count += 1
                            all_count += 1
                            pbar.set_description(
                                f"Filtered: {filter_count:,}/{all_count:,}[{filter_count / all_count:.2%}]")


if __name__ == "__main__":
    sp = SmilesPreprocessor()
    sp.prep_uspto()
