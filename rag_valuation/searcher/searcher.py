import scann
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util


class RagSearcher:
    def __init__(self, csv_path, model_name="BAAI/bge-large-en-v1.5", embeddings_path=None):
        self.df = pd.read_csv(csv_path)
        self.text_list = self.df['chunk_text'].tolist()
        self.text_list_sources = self.df['source_org'].tolist()

        self.model = SentenceTransformer(model_name)

        if embeddings_path:
            self.embeddings = torch.load(embeddings_path)
            self._prepare_searcher()
        else:
            self.embeddings = None

    def _prepare_searcher(self):

        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1)[:, np.newaxis]
        self.searcher = scann.scann_ops_pybind.builder(normalized_embeddings, 20, "dot_product").tree(
            num_leaves=360, num_leaves_to_search=36, training_sample_size=250000).score_ah(
            2, anisotropic_quantization_threshold=0.2).reorder(150).build()

    def create_embeddings(self):
        self.embeddings = self.model.encode(self.text_list, convert_to_tensor=True)
        self._prepare_searcher()

    def query(self, q: str):
        query_embedding = self.model.encode([q])[0]
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        neighbors, distances = self.searcher.search(normalized_query, final_num_neighbors=20)

        return pd.DataFrame({
            'index': [neighbors[i] for i in range(len(neighbors))],
            'distance': [distances[i] for i in range(len(neighbors))],
            'text': [self.text_list[neighbors[i]] for i in range(len(neighbors))],
            'source': [self.text_list_sources[neighbors[i]] for i in range(len(neighbors))]
        })