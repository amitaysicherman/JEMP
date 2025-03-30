from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm

batch_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D", trust_remote_code=True)
esm = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D", trust_remote_code=True)
esm.eval().to(device)
for param in esm.parameters():
    param.requires_grad = False
for split in ["train", "test"]:
    base_dir = "data/CARE_datasets/hard"
    data_file = f"{base_dir}/{split}_enzyme.txt"
    output_file = f"{base_dir}/{split}_enzyme.np"
    embeddings = []
    with open(data_file) as f:
        lines = f.readlines()

    for i in tqdm(range(0, len(lines), batch_size)):
        batch = lines[i:i + batch_size]
        tokens = tokenizer(batch, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = esm(input_ids=tokens["input_ids"].to(device),
                          attention_mask=tokens["attention_mask"].to(device))
        encoder_outputs = outputs.last_hidden_state.mean(axis=1).detach().cpu().numpy().tolist()
        embeddings.extend(encoder_outputs)
    np.save(output_file, np.array(embeddings))
    print(f"Finished processing {split}")
