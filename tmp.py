import torch
from transformers import T5Config, T5ForConditionalGeneration
from tqdm import tqdm


class Trie:
    def __init__(self, tokens_lists, empty_token=0):
        self.root = {}
        self.empty_token = empty_token
        self.nodes_count = 0
        pbar = tqdm(tokens_lists)
        for word_list in pbar:
            self._insert(word_list)
            pbar.set_description(f"Number of nodes in the trie: {self.nodes_count}")

    def _insert(self, word_list):
        current = self.root
        for num in word_list:
            if num not in current:
                current[num] = {}
                self.nodes_count += 1
            current = current[num]

    def search_prefix(self, prefix):
        current = self.root
        for num in prefix:
            if num not in current:
                return [self.empty_token]
            current = current[num]
        if len(current) == 0:
            return [self.empty_token]
        return list(current.keys())

    def print_trie(self, tokenizer=None):
        def print_node(node, depth=0):
            for key, value in node.items():
                if tokenizer is not None:
                    print("  " * depth + tokenizer.decode([key]))
                else:
                    print("  " * depth + str(key))
                print_node(value, depth + 1)

        print_node(self.root)


def build_trie(word_list, tokenizer, decoder_start_token):
    tokens_list = []
    for word in word_list:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        tokens = [decoder_start_token] + tokens + [tokenizer.eos_token_id]
        tokens_list.append(tokens)
    return Trie(tokens_list, empty_token=tokenizer.pad_token_id)


# if name main:
if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D", trust_remote_code=True)
    tgt_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

    tokenizer.vocab = {**tokenizer.get_vocab(), **tgt_tokenizer.get_vocab()}

    config = T5Config(
        vocab_size=len(tokenizer.vocab),
        n_positions=512,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_layers=6,
        decoder_start_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        is_encoder_decoder=True,
        bos_token_id=tokenizer.bos_token_id,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T5ForConditionalGeneration(config)
    cp_file = "JERP/model.safetensors"
    from safetensors.torch import load_file

    missing_keys = model.load_state_dict(load_file(cp_file), strict=False)
    # print(f"Missing keys: {missing_keys}")
    with open('data/Reactzyme/data/test_enzyme.txt') as f:
        proteins_list = f.read().splitlines()
    trie = build_trie(proteins_list, tokenizer, tokenizer.pad_token_id)

    prefix_allowed_tokens_fn = lambda batch_id, sent: trie.search_prefix(sent.tolist())
    prefix_input = "CNC(CCCC[N+](C)(C)C)C(C)=O.CNC(CCCC[NH3+])C(C)=O.C[S+](CCC([NH3+])C(=O)[O-])CC1OC(n2cnc3c(N)ncnc32)C(O)C1O.Nc1ncnc2c1ncn2C1OC(CSCCC([NH3+])C(=O)[O-])C(O)C1O"
    prefix = tokenizer(prefix_input, return_tensors="pt", max_length=1024,
                       padding="max_length", truncation=True)
    gt = "MSEIRLYVTTTMSEIRLYVTTTEAKAEEILDLLSALFGEEDFAIGTTEIDEKKDIWEASIYMMAEDEAEVQSRVEDALKASFPDARLEREVIPEIDWVVKSLEGLKPVRAGRFLVHGSHDRDKIRPGDIAIEIDAGQAFGTGHHGTTAGCLEVIDSVVRSRPVRNALDLGTGSGVLAIAVRKLRNIPVLATDIDPIATKVAAENVRRNGIASGIVTRTAPGFHSTAFSEHGPFDLIIANILARPLIRMAPKLATHLAPGGSVILSGILAGQRWKVIAAYSGARLRHVKTIWRNGWVTIHLDRP"
    labels = tokenizer(gt, return_tensors="pt", max_length=1024, padding="max_length", truncation=True)['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100
    prefix = {k: v.to(device) for k, v in prefix.items()}
    labels = labels.to(device)
    output_predict = model(**prefix, labels=labels)['logits'].argmax(dim=-1)
    print(output_predict[0])
    print(f"Predicted: {tokenizer.decode(output_predict[0])}")


    output = model.generate(**prefix,
                            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, num_beams=10, num_return_sequences=10,
                            max_length=1024, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                            decoder_start_token_id=tokenizer.pad_token_id)

    ouput_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    for i, out in enumerate(ouput_decoded):
        print(f"Output {i}: {out.replace(' ', '')}",out.replace(' ', '') == gt)
