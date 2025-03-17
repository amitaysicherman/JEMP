import torch
import os
def load_state_dict_from_last_cp(base_dir):
    all_cp_dirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
    if len(all_cp_dirs) == 0:
        return None
    all_cp_dirs.sort(key=lambda x: int(x.split("-")[1]), reverse=True)

    last_cp_dir = all_cp_dirs[0]
    last_cp_path = os.path.join(base_dir, last_cp_dir, "pytorch_model.bin")
    state_dict = torch.load(last_cp_path, map_location="cpu")
    return state_dict



