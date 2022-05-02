import torch
import h5py
from types import Tuple
from train import DualEncoder

def load_model_from_checkpoint(checkpoint_path: Path) -> DualEncoder:
    model = DualEncoder.load_from_checkpoint(checkpoint_path)
    return model

def load_data(data_file: Path) -> Tuple[List[str], List[str]]:
    with h5py.File(data_file, "r") as fp:
        queries = fp["queries"].asstr()
        docs = fp["docs"].asstr()
    return queries, docs

def creata_doc_index(model: DualEncoder, queries: List[str], docs: List[str]) -> None:
    pass

def main():
    # TODO: load model
    # TODO: load data
    # TODO: load Make Index
    pass

if __name__ == "__main__":
    main()