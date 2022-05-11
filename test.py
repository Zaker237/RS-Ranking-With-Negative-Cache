import os
from pyserini.dsearch import SimpleDenseSearcher, TctColBertQueryEncoder

from config import MODEL_PATH, DOCS_PATH, INDEX_PATH

TEXT_DATA_FILE = "./msmarco-test2019-queries.tsv"
RESULT_FILE = "./msmarco-test2019-results.tsv"



def main():
    test_data = load_tsv_file(TEXT_DATA_FILE)
    

    encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
    searcher = SimpleDenseSearcher.from_prebuilt_index(
        'msmarco-passage-tct_colbert-hnsw',
        encoder
    )
    result_data = []
    for data in test_data:
        hits = searcher.search(data[1])

        for i in range(0, len(hits)):
            result_string = f"{data[0]} Q0 D{hits[i].docid:7} {i+1} {hits[i].score:.5f} IndriQueryLikelihood"
            result_data.append(result_string)
            print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')

    save_file(result_data, RESULT_FILE)

    print("______End________")
        

if __name__ == "__main__":
    main()