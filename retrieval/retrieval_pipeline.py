import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class RetrievalPipeline:
    def __init__(self, corpus):
        self.tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dense_index = faiss.IndexFlatL2(384)
        self.build_dense_index(corpus)
    
    def build_dense_index(self, corpus):
        embeddings = self.encoder.encode(corpus, convert_to_tensor=False)
        self.dense_index.add(embeddings)

    def retrieve(self, query, top_n=5):
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_results = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_n]

        query_embedding = self.encoder.encode([query], convert_to_tensor=False)
        _, dense_results = self.dense_index.search(query_embedding, top_n)

        return {'bm25': top_bm25_results, 'dense': dense_results[0].tolist()}

if __name__ == '__main__':
    sample_corpus = ["Patient has diabetes and needs medication recommendations.",
                     "Hypertension case with heart complications."]
    pipeline = RetrievalPipeline(sample_corpus)
    print(pipeline.retrieve("Diabetes treatment"))
