import json
from collections import OrderedDict
from datasets import load_dataset
from collabllm.datasets.dataset import ChatDataset


class TwentyQMT(ChatDataset):
    def __init__(self, repo_id='aditijb/collabllm-20q', split='test'):
        """
        Initializes the 20Q multiturn dataset by loading and preprocessing the data.
        """
        print(f"Loading 20Q multiturn data from HF: {repo_id} [{split}]")
        self.split = split  # aditi addition for later use in preprocess
        raw_data = load_dataset(repo_id, split=split, trust_remote_code=True)
        processed_data = self.preprocess(raw_data)
        super().__init__(processed_data)

    def preprocess(self, raw_data):
        processed_data = []
        for entry in raw_data:
            metadata = {
                "id": entry.get("idx"),
                "target": entry.get("prompt_item"),
                "metrics": entry.get("chosen_eval", {}),
                "split": self.split,  # added by aditi to fix that dataset.py's to_hf_dataset() requires the splot of train vs test
            }
            turns = entry["prompt"]
            turns.append({"role": "assistant", "content": entry["chosen"]})
            processed_data.append({"metadata": metadata, "chat": turns})
        return processed_data


def load_single_turn_20q_dataset(repo_id='aditijb/collabllm-20q', split='test', n_eval=None):
    """
    Loads and returns a single-turn version of the 20Q dataset, where each example contains the
    last user-assistant turn pair.
    """
    raw_data = load_dataset(repo_id, split=split)
    if n_eval:
        raw_data = raw_data.shuffle(seed=42).select(range(n_eval))

    single_turn_data = []
    for ex in raw_data:
        chat = ex["prompt"]
        if len(chat) >= 2 and chat[-1]["role"] == "assistant" and chat[-2]["role"] == "user":
            single_turn_data.append({
                "chat": [chat[-2], chat[-1]],
                "id": ex.get("idx"),
                "target": ex.get("prompt_item"),
                "metrics": ex.get("chosen_eval", {}),
            })
    return single_turn_data
