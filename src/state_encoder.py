import torch
from transformers import AutoTokenizer, AutoModel

class StateEncoder:
    """
    Encodes story sentences into dense vectors using DistilBERT.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # inference mode

    def encode(self, text: str) -> torch.Tensor:
        """
        Returns a 768-dim embedding for the input sentence.
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=64
            )
            outputs = self.model(**inputs)
            # mean-pool over tokens
            embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.squeeze(0)
