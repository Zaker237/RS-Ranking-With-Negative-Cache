import torch
import h5py
from typing import List, Tuple
from pathlib import Path
from model import DualEncoder

def load_model_from_checkpoint(checkpoint_path: Path) -> DualEncoder:
    model = DualEncoder.load_from_checkpoint(checkpoint_path)
    return model

def load_docs(data_file: Path) -> List[str]:
    with h5py.File(data_file, "r") as fp:
        # queries = fp["queries"].asstr()
        docs = fp["docs"].asstr()
    return docs

def creata_doc_index(model: DualEncoder, queries: List[str], docs: List[str]) -> None:
    pass

def main():
    # TODO: load model
    # TODO: load data
    # TODO: load Make Index
    pass

if __name__ == "__main__":
    main()