from typing import List, Dict, Any, Optional

from src.embedding import embed_texts

try:
    from config import TOP_K
except Exception:
    TOP_K = 5


SIMILARITY_THRESHOLD = 0.28
MIN_RESULTS_REQUIRED = 2


# EMBEDDING
def embed_query(question: str, model) -> List[float]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    try:
        embedding = embed_texts(model, [question])
        return embedding[0].tolist()

    except Exception as e:
        raise RuntimeError(f"Failed to embed query: {e}")

# RETRIEVAL
def retrieve(
    question: str,
    collection,
    model,
    k: Optional[int] = None,
    where: Optional[dict] = None,
) -> List[Dict[str, Any]]:

    k = k or TOP_K

    try:
        query_embedding = embed_query(question, model)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        if not results:
            return []

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        meta_list = results.get("metadatas", [[]])[0]
        dist_list = results.get("distances", [[]])[0]

        if not ids_list:
            return []

        retrieved = []

        for chunk_id, doc, meta, dist in zip(ids_list, docs_list, meta_list, dist_list):
            try:
                similarity = 1.0 / (1.0 + float(dist))

                retrieved.append({
                    "chunk_id": chunk_id,
                    "chunk_text": doc or "",
                    "metadata": meta or {},
                    "similarity": round(similarity, 4),
                })

            except Exception:
                continue

        retrieved.sort(key=lambda x: x["similarity"], reverse=True)

        if not retrieved:
            return []

        top_score = retrieved[0]["similarity"]

        filtered = [
            r for r in retrieved
            if r["similarity"] >= max(0.20, top_score * 0.75)
        ]

        if len(filtered) < 2:
            filtered = retrieved[:2]

        return filtered

    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return []