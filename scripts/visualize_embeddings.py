"""
Visualize the knowledge base's embedding space in 2D with t-SNE.

Embeddings have thousands of dimensions, far too many to plot. t-SNE compresses them to
2D while trying to keep points that were close together in the original space close
together on the plot. If the embeddings capture meaning, chunks on the same topic (or
from the same doc_type) should cluster. This is a diagnostic only — retrieval itself
always happens in the full-dimensional space inside ChromaDB.

Reads the index built by build_index.py and writes docs/embedding_space.png.

Run from the project root:
  python -m scripts.visualize_embeddings
"""

import os

import chromadb
import matplotlib

matplotlib.use("Agg")  # render to a file; no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from rag.config import CHROMA_PATH, COLLECTION_NAME

OUTPUT_IMAGE = os.path.join("docs", "embedding_space.png")
MIN_POINTS = 4  # t-SNE needs a handful of points to say anything useful

# Fixed colours for the known categories; any new doc_type gets a neutral grey.
DOC_TYPE_COLORS = {"profile": "#4C72B0", "projects": "#DD8452", "skills": "#55A868"}
FALLBACK_COLOR = "#777777"


def main() -> None:
    if not os.path.exists(CHROMA_PATH):
        raise SystemExit(f"No vector store at ./{CHROMA_PATH}/. Run `python build_index.py` first.")

    collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION_NAME)
    data = collection.get(include=["embeddings", "metadatas", "documents"])
    vectors = np.asarray(data["embeddings"])
    metadatas, documents = data["metadatas"], data["documents"]
    n_points = len(vectors)

    print(f"Loaded {n_points} chunk embeddings.")
    if n_points < MIN_POINTS:
        raise SystemExit(f"Need at least {MIN_POINTS} chunks for a meaningful plot.")

    # Perplexity (roughly: neighbours considered per point) must be below the number of
    # points, and a small value is more stable for a small knowledge base.
    perplexity = max(2, min(5, n_points - 1))
    print(f"Running t-SNE (perplexity={perplexity})...")
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca").fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(10, 7))
    doc_types = sorted({m["doc_type"] for m in metadatas})
    for doc_type in doc_types:
        idx = [i for i, m in enumerate(metadatas) if m["doc_type"] == doc_type]
        ax.scatter(
            coords[idx, 0], coords[idx, 1], label=doc_type, s=120, alpha=0.8,
            color=DOC_TYPE_COLORS.get(doc_type, FALLBACK_COLOR),
            edgecolors="white", linewidths=1.5,
        )
    for i in range(n_points):
        preview = documents[i][:40].replace("\n", " ") + "..."
        ax.annotate(preview, coords[i], fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_title("Knowledge base embedding space (t-SNE projection)")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(title="doc_type")
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"Saved {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
