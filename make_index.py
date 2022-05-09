import faiss
import h5py
from typing import List, Tuple
from pathlib import Path
from model import DualEncoder
from transformers import AutoTokenizer

MODEL_PATH = Path("/home/mboutchouang/test_negative_cache/lightning_logs/version_4/checkpoints/epoch=15-step=353952.ckpt")
DOCS_PATH = Path("/home/mboutchouang/ranking-utils/outputs/2022-04-11/11-32-20/data.h5")
INDEX_PATH = Path("./indexes")


class Indexes():
    def __init__(self, model: DualEncoder, bert_model: str):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        # self.config = faiss.GpuIndexConfig()
        # self.config.useFloat16 = False
        # self.config.device = 2
        # self.resource = faiss.StandardGpuResources()
        # self.index = faiss.GpuIndexFlatL2(
        #     self.resource,
        #     self.model.document_encoder.config.hidden_size,
        #     self.config
        # )
        self.index = faiss.IndexFlatL2(self.model.document_encoder.config.hidden_size)

    def load_index(self, path: Path):
        # load the faiss index from the disk
        self.index = faiss.read_index(str(path))
        print("Index loaded from", path)

    def save_index(self, path: Path):
        # save the faiss index on the disk
        faiss.write_index(self.index, str(path))
        print("Index saved to", path)

    def add_doc_to_index(self, doc: str):
        doc_data = self.tokenizer([doc], padding=True, truncation=True, return_tensors="pt")
        embedding = self.model.encode_doc(doc_data).detach().cpu().numpy()
        self.index.add(embedding)


def load_docs(data_file: Path) -> List[str]:
    with h5py.File(data_file, "r") as fp:
        queries = fp["queries"].asstr()
        docs = fp["docs"].asstr()
        for doc in docs:
            yield doc


def load_model_from_checkpoint(checkpoint_path: Path) -> DualEncoder:
    model = DualEncoder.load_from_checkpoint(checkpoint_path)
    print("Model loaded")
    return model


def main():
    # TODO: load model
    model = load_model_from_checkpoint(MODEL_PATH)

    # Init the index encoder
    index = Indexes(model, "bert-base-uncased")
    print("Index initialized", index.index.is_trained)

    # TODO: load data and build index
    i = 0
    for document in load_docs(DOCS_PATH):
        print(f"Adding document {i}")
        index.add_doc_to_index(document)
        i += 1

    # the Inedex is now ready to be used and saved
    index.save_index(INDEX_PATH)


if __name__ == "__main__":
    main()