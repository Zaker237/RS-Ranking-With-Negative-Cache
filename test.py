import os
import faiss
import argparse
import h5py
from sklearn.utils import resample
from make_index import Indexes
from config import MODEL_PATH, DOCS_PATH, INDEX_PATH
from model import DualEncoder
from transformers import AutoTokenizer
from pathlib import Path
from typing import List


def load_queries(data_file: Path) -> List[str]:
    with open(data_file, "r", encoding="utf-8") as fp:
        data = fp.readlines()

    return list(map(lambda x: tuple(x.replace("\n", "").split("\t")) , data))

def encore_queries(query: str, model: DualEncoder, tokenizer: AutoTokenizer):
    query_data = tokenizer([query], padding=True, truncation=True, return_tensors="pt")
    embedding = model.encode_query(query_data).detach().cpu().numpy()
    return embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_data", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    args = parser.parse_args()

    index = faiss.read_index(str(Path(args.index_dir, "index")))

    model = DualEncoder.load_from_checkpoint(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    queries = load_queries(Path(args.query_data))
    results = []
    for idx, query in queries:
        embedding = encore_queries(query, model, tokenizer)
        D, I = index.search(embedding, k=100)
        for j, (d, i) in enumerate(zip(list(D[0]), list(I[0]))):
            result_string = f"{idx} Q0 {i} {j+1} {d} DualEncoder"
            results.append(result_string)

    with open(args.output_file, "w") as fp:
        fp.write("\n".join(results))
    

if __name__ == "__main__":
    main()