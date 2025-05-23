from datasets import load_dataset
from collabllm.datasets.dataset import ChatDataset

class TwentyQMT(ChatDataset):
    def __init__(self, repo_id='aditijb/collabllm-20q', split='test'):
        print(f"Loading 20Q single-turn data from HF: {repo_id} [{split}]")
        self.split = split
        raw_data = load_dataset(repo_id, split=split, trust_remote_code=True)

        # Extract only single-turn user-assistant pairs
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

        super().__init__(single_turn_data)
