from tqdm import tqdm

import torch

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

def build_mask_from_trie(trie, sequences, vocab_size):
    """
    Generate a mask tensor indicating valid next tokens based on a trie and input sequences.

    Args:
        trie (Trie): The trie structure where each node represents a token and contains child nodes.
        sequences (torch.Tensor): Tensor of shape (batch_size, seq_length) containing token indices.
        vocab_size (int): The size of the vocabulary.

    Returns:
        torch.Tensor: A mask tensor of shape (batch_size, seq_length, vocab_size) with 1s indicating valid next tokens.
    """
    batch_size, seq_length = sequences.shape
    mask = torch.zeros((batch_size, seq_length, vocab_size), dtype=torch.float)
    for batch_idx in range(batch_size):
        current_node = trie.root
        for seq_idx in range(seq_length):
            token = int(sequences[batch_idx, seq_idx].item())
            current_node = current_node[token]
            for child_token in current_node.keys():
                # Set valid next tokens to 1 in the mask
                mask[batch_idx, seq_idx, child_token] = 1
    return mask


