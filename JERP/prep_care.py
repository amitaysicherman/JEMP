import os

import pandas as pd
import numpy as np
import random
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')


def remove_stereo_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def remove_stereo(rnx):
    src, tgt = rnx.split(">>")
    src = src.split(".")
    tgt = tgt.split(".")
    src = [remove_stereo_mol(s) for s in src]
    tgt = [remove_stereo_mol(t) for t in tgt]
    src = ".".join(src)
    tgt = ".".join(tgt)
    return src + ">>" + tgt


class EC2Protein:
    def __init__(self, mapping_file="data/CARE_datasets/processed_data/protein2EC.csv", seed=42):
        self.mapping_file = mapping_file
        self.df = pd.read_csv(mapping_file)
        random.seed(seed)
        self.ec_to_protein = self.df.groupby('EC number')['Sequence'].apply(lambda x: random.choice(list(x))).to_dict()

    def get_protein(self, ec_number):
        """
        Given an EC number, return a random protein sequence associated with it.
        """
        if ec_number in self.ec_to_protein:
            return self.ec_to_protein[ec_number]
        else:
            raise ValueError(f"EC number {ec_number} not found in the mapping file.")


if __name__ == "__main__":

    ec2protein = EC2Protein()
    for level in ["easy", "medium", "hard"]:
        output_base = f"data/CARE_datasets/{level}/"
        os.makedirs(output_base, exist_ok=True)
        for split in ["train", "test"]:
            data = pd.read_csv(f"data/CARE_datasets/splits/task2/{level}_reaction_{split}.csv")
            reactions = data["Reaction"].tolist()
            reactions = [remove_stereo(r) for r in tqdm(reactions)]
            ec = data["EC number"].tolist()
            ec = [ec2protein.get_protein(e) for e in tqdm(ec)]
            with open(f"{output_base}/{split}_reaction.txt", "w") as reaction_file:
                with open(f"{output_base}/{split}_enzyme.txt", "w") as enzyme_file:
                    for r, e in zip(reactions, ec):
                        reaction_file.write(r + "\n")
                        enzyme_file.write(e + "\n")
    # ec2protein = EC2Protein(mapping_file="/Users/amitay.s/Downloads/14004425/CARE_datasets/processed_data/protein2EC.csv")
    # ec2protein = EC2Protein(mapping_file="data/CARE_datasets/processed_data/protein2EC.csv")
