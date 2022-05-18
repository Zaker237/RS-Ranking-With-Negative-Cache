import faiss
import torch
import h5py
from typing import List, Tuple
from pathlib import Path
from model import DualEncoder
from transformers import AutoTokenizer
from config import MODEL_PATH, DOCS_PATH, INDEX_PATH, INDEX_ID_PATH


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
        self.docid = []

    def load_index(self, path: Path):
        # load the faiss index from the disk
        self.index = faiss.read_index(str(path))
        print("Index loaded from", path)

    def save_index(self, path: Path):
        # save the faiss index on the disk
        faiss.write_index(self.index, str(path))
        print("Index saved to", path)

    def save_docid(self, path: Path):
        with open(str(path), "w") as fp:
            fp.write("\n".join(self.docid))

    def add_doc_to_index(self, doc: str):
        doc_data = self.tokenizer([doc], padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        embedding = self.model.encode_doc(doc_data).detach().cpu().numpy()
        self.index.add(embedding)


def load_docs(data_file: Path) -> List[str]:
    docs = []
    with h5py.File(data_file, "r") as fp:
        # queries = fp["queries"].asstr()
        docs = fp["docs"].asstr()
        docs_ids = fp["orig_doc_ids"].asstr()
        for doc, doc_id in zip(docs, docs_ids):
            yield doc, doc_id


def load_model_from_checkpoint(checkpoint_path: Path) -> DualEncoder:
    model = DualEncoder.load_from_checkpoint(checkpoint_path)
    print("Model loaded")
    return model


def main():
    # TODO: load model
    device = torch.device('cuda')
    model = load_model_from_checkpoint(MODEL_PATH).to(device)

    # Init the index encoder
    index = Indexes(model, "bert-base-uncased")
    print("Index initialized", index.index.is_trained)

    # TODO: load data and build index
    
    for document, document_id in load_docs(DOCS_PATH):
        index.add_doc_to_index(document, document_id)
        print(f"Adding document {i}")
        index.docid.append(str(document_id))
    
    index.save_index(INDEX_PATH)
    index.save_docid(INDEX_ID_PATH)


if __name__ == "__main__":
    main()
