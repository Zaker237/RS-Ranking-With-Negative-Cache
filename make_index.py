import torch
import h5py
from typing import List, Tuple
from pathlib import Path
from model import DualEncoder

MODEL_PATH = Path("/home/mboutchouang/ranking-utils/outputs/2022-04-11/11-32-20/checkpoint_epoch=16.ckpt")
DOCS_PATH = Path("/home/mboutchouang/ranking-utils/outputs/2022-04-11/11-32-20/data.h5")
INDEX_PATH = Path("./indexes")

class Document(object):
    ID: int = 0
    def __init__(self, doc: str):
        self.ID += 1
        self.doc_id = self.ID
        self.doc = doc


    @property
    def query(self):
        return self.doc


class Indexs():
    def __init__(self):
        self.docs_ids = []
        self.docs_indexes = {}

    def index_documents(self, model: DualEncoder, document: Document):
        if document.doc_id not in self.docs_ids:
            self.docs_ids.append(document.doc_id)
            self.docs_ids[document.doc_id] = model(document.query, document.document)

    def save_index(self, path: Path):
        with open(path + "/docid", "w") as fp:
            fp.write("\n".join(map(str, self.docs_ids)))

        with open(path + "/index", "wr") as fp:
            for doc_id in self.docs_ids:
                fp.write("\n".join(map(str, self.docs_indexes[doc_id])))


def load_docs(data_file: Path) -> List[str]:
    with h5py.File(data_file, "r") as fp:
        # queries = fp["queries"].asstr()
        docs = fp["docs"].asstr()
        yield Document(docs)


def load_model_from_checkpoint(checkpoint_path: Path) -> DualEncoder:
    model = DualEncoder.load_from_checkpoint(checkpoint_path)
    return model



def main():
    # TODO: load model
    model = load_model_from_checkpoint(MODEL_PATH)

    # Init the index

    indexes = Indexs()

    # TODO: load data and build index
    for document in load_docs(DOCS_PATH):
        indexes.index_documents(model, document)

    indexes.save_index(INDEX_PATH)

if __name__ == "__main__":
    main()