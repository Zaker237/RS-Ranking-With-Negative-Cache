import os
from pathlib import Path

MODEL_PATH = Path("/home/mboutchouang/test_negative_cache/lightning_logs/version_4/checkpoints/epoch=15-step=353952.ckpt")
DOCS_PATH = Path("/home/mboutchouang/ranking-utils/outputs/2022-04-11/11-32-20/data.h5")
INDEX_PATH = Path("./msmarco_index/index")
INDEX_ID_PATH = Path("./msmarco_index/docid")