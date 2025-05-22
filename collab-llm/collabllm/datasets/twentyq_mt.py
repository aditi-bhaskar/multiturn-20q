import json
from collections import OrderedDict
from datasets import load_dataset
from collabllm.datasets.dataset import ChatDataset


class TwentyQMT(ChatDataset):

    def __init__(self, repo_id='aditijb/collabllm-20q', split='dev'):
        """
        Initializes the 20q dataset with raw data.

        Parameters:
        raw_data (dict): The raw 20q data to be processed (public repo on hf)
        """
        print(f"Loading 20Q multiturn data from HF: {repo_id} [{split}]")
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
            }
            # Ensure prompt is a valid list of {"role": ..., "content": ...}
            turns = entry["prompt"]
            turns.append({"role": "assistant", "content": entry["chosen"]})
            processed_data.append({"metadata": metadata, "chat": turns})
        return processed_data
        