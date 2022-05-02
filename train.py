import h5py
from transformers import AutoTokenizer
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from model import DualEncoder

class TrainingDataset(Dataset):
    def __init__(
        self,
        data_file: Path,
        train_file: Path,
        bert_model: str
    ) -> None:
        super().__init__()
        self.data_file = data_file
        self.train_file = train_file
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

    def __len__(self) -> int:
        with h5py.File(self.train_file, "r") as fp:
            return len(fp["q_ids"])

    def __getitem__(self, index: int):
        with h5py.File(self.train_file, "r") as fp:
            q_id = fp["q_ids"][index]
            pos_doc_id = fp["pos_doc_ids"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            pos_doc = fp["docs"].asstr()[pos_doc_id]
        return query, pos_doc
    
    def collate_fn(self, inputs):
        queries, docs = zip(*inputs)
        q_inputs = self.tokenizer(list(queries), padding=True, truncation=True, return_tensors="pt")
        d_inputs = self.tokenizer(list(docs), padding=True, truncation=True, return_tensors="pt")
        return q_inputs, d_inputs


def main():
    pl.seed_everything(123, workers=True)

    h5_dir = Path("/home/mboutchouang/ranking-utils/outputs/2022-04-11/11-32-20")
    ds = TrainingDataset(h5_dir / "data.h5", h5_dir / "fold_0" / "train_pairwise.h5", "bert-base-uncased")
    dl = DataLoader(ds, batch_size=6, collate_fn=ds.collate_fn, num_workers=16)
    model = DualEncoder(lr=1e-5)
    trainer = pl.Trainer(precision=16, accelerator="gpu", gpus=[0, 1, 2, 3], strategy="ddp", max_epochs=16)
    trainer.fit(model=model, train_dataloaders=dl)

if __name__ == "__main__":
    main()