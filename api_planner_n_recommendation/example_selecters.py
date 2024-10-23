import os, sys
import numpy as np
import pandas as pd
from models import BaseEmbModel
from pathlib import Path
from typing import Union
import os
import faiss

import logging

class ExampleSelecter():
    """
    Args:
    - example_bank_path: Path, the path of the example bank json file.
    - emb_field: str, the field name in example bank to be embedded as index.
    - emb_model: BaseEmbModel, the embedding model.
    - examplebank_emb_path: Path, the path to save or load the index of the example bank in .npy file.
    """
    def __init__(self, example_bank_path:Path, emb_field:str, emb_model:BaseEmbModel, examplebank_emb_path:Path=None):
        self.example_bank_path = example_bank_path
        self.emb_field = emb_field
        self.examplebank_emb_path = examplebank_emb_path
        self.emb_model = emb_model

    def build_example_emb_pool(self):
        if self.examplebank_emb_path == None:
            self.examplebank_emb_path = self.example_bank_path.parent / Path('examplebank_emb.npy')
        example_bank = pd.read_json(self.example_bank_path)
        example_emb_li = []
        for i, item in example_bank.iterrows():
            emb = self.emb_model.get_emb(text=item[self.emb_field])
            example_emb_li.append(emb)
        xb = np.array(example_emb_li)
        logging.info(f"built api embeddings of shape: `{xb.shape}`, embedding field: {self.emb_field}, embedding model: {self.emb_field.model_name}. The embedding pool is saved at {self.apibank_emb_path}")
        np.save(self.examplebank_emb_path, xb)
    
    def load_examples(self):
        if self.examplebank_emb_path == None:
            self.examplebank_emb_path = self.example_bank_path.parent / Path('examplebank_emb.npy')
        if not os.path.exists(self.examplebank_emb_path):
            raise FileNotFoundError(f"embedding pool not found at {self.examplebank_emb_path}.")
        if not os.path.exists(self.example_bank_path):
            raise FileNotFoundError(f"example bank not found at {self.example_bank_path}.")
        self.example_bank = pd.read_json(self.example_bank_path)
        self.examplebank_emb = np.load(self.examplebank_emb_path).astype('float32')
        return self

    def select_examples(self, user_req:Union[str,np.ndarray], k:int=4)->pd.DataFrame:
        if self.example_bank is None or self.examplebank_emb is None:
            raise Exception("Please load examples and their embeddings using `load_examples` first.")
        if isinstance(user_req, str):
            user_emb = np.array(self.emb_model.get_emb(user_req), dtype='float32')
        elif isinstance(user_req, np.ndarray):
            user_emb = user_req

        d = self.examplebank_emb.shape[1]     
        index = faiss.IndexFlatIP(d)   

        xb = self.examplebank_emb.copy()/np.linalg.norm(self.examplebank_emb)
        faiss.normalize_L2(xb)
        index.add(xb)         
        logging.debug("faiss indexed item number:", index.ntotal)
            
        xq = [user_emb].copy()/np.linalg.norm([user_emb])
        faiss.normalize_L2(xq)

        D, I = index.search(xq, k)
        logging.info(f"top-{k} index:\n{I}")
        logging.info(f"top-{k} similarity:\n{D}")

        return self.example_bank.iloc[I[0]]