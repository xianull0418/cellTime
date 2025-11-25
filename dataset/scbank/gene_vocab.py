import json
from typing import Dict, List, Union, Mapping
from pathlib import Path

class GeneVocab(dict):
    """
    A simple GeneVocab implementation to replace scgpt.tokenizer.GeneVocab
    """
    def __init__(self, vocab: Dict[str, int]):
        super().__init__(vocab)
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "GeneVocab":
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if file_path.suffix in [".tsv", ".txt"]:
            return cls.from_list_file(file_path)
            
        try:
            with open(file_path, "r") as f:
                vocab = json.load(f)
            return cls(vocab)
        except json.JSONDecodeError:
            # Fallback to trying to read as list file if JSON fails
            return cls.from_list_file(file_path)

    @classmethod
    def from_list_file(cls, file_path: Union[str, Path]) -> "GeneVocab":
        """Load vocabulary from a file with one gene per line"""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        vocab = {}
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                token = line.strip()
                if token:
                    vocab[token] = i
        return cls(vocab)

    @classmethod
    def from_dict(cls, vocab: Dict[str, int]) -> "GeneVocab":
        return cls(vocab)

    def save_json(self, file_path: Union[str, Path]) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with open(file_path, "w") as f:
            json.dump(self.vocab, f, indent=2)
            
    @property
    def pad_token(self):
        return "<pad>"
        
    @property
    def pad_token_id(self):
        return self.get(self.pad_token, None)
