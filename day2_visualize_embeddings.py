"""
Day 2 — Visualize the embedding space with t-SNE.

Embeddings from gemini-embedding-001 have thousands of dimensions — way too many to
plot directly. t-SNE (t-distributed Stochastic Neighbor Embedding) compresses
high-dimensional vectors down to 2 dimensions while trying to preserve which points
were close together in the original space. The result is a scatter plot where chunks
with similar meaning should cluster near each other.

This is purely for understanding/demonstration — it's not part of the retrieval
pipeline itself (retrieval still happens in the full-dimensional space, via Chroma).
It's a genuinely useful thing to show in an interview or a README: a picture that
proves the embeddings are actually capturing meaningful structure, not just noise.

Run day2_build_vector_store.py first to create the vector store this script reads.

Run:
  python day2_visualize_embeddings.py
"""

import chromadb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "ashutosh_knowledge_base"
OUTPUT_IMAGE = "embedding_space.png"

# One distinct color per doc_type, so clusters are easy to read at a glance.
DOC_TYPE_COLORS = {
    "profile": "#4C72B0",
    "projects": "#DD8452",
    "skills": "#55A868",
}


def main():
    print("Loading vector store...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    data = collection.get(include=["embeddings", "metadatas", "documents"])
    vectors = data["embeddings"]
    metadatas = data["metadatas"]
    documents = data["documents"]
    n_points = len(vectors)

    print(f"Loaded {n_points} chunk embeddings.")
    if n_points < 4:
        print(
            "Need at least 4 chunks for a meaningful t-SNE plot — "
            "add more documents to your knowledge base first."
        )
        return

    # t-SNE's 'perplexity' roughly controls how many neighbors each point considers.
    # It must be smaller than the number of points, and with a small knowledge base
    # like ours (a few dozen chunks) a low value gives more stable results than
    # sklearn's default of 30.
    perplexity = max(2, min(5, n_points - 1))
    print(f"Running t-SNE (perplexity={perplexity})...")

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    coords_2d = tsne.fit_transform(vectors.__array__() if hasattr(vectors, "__array__") else vectors)

    print("Plotting...")
    fig, ax = plt.subplots(figsize=(10, 7))

    for doc_type in sorted(set(m["doc_type"] for m in metadatas)):
        xs = [coords_2d[i][0] for i in range(n_points) if metadatas[i]["doc_type"] == doc_type]
        ys = [coords_2d[i][1] for i in range(n_points) if metadatas[i]["doc_type"] == doc_type]
        ax.scatter(
            xs, ys,
            label=doc_type,
            color=DOC_TYPE_COLORS.get(doc_type, "#777777"),
            s=120,
            alpha=0.8,
            edgecolors="white",
            linewidths=1.5,
        )

    # Label each point with a short preview of its text, so the plot is readable
    # on its own without needing the raw data alongside it.
    for i in range(n_points):
        preview = documents[i][:40].replace("\n", " ") + "..."
        ax.annotate(
            preview,
            (coords_2d[i][0], coords_2d[i][1]),
            fontsize=7,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_title("Knowledge Base Embedding Space (t-SNE projection)")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(title="doc_type")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"\nSaved plot to {OUTPUT_IMAGE} — open it to view.")


if __name__ == "__main__":
    main()
