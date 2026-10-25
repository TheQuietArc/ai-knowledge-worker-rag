"""Document loading and chunking."""

import os

import pytest

from rag.ingestion import load_and_chunk_documents

DOCS_FOLDER = "data/raw_docs"


@pytest.fixture(scope="module")
def chunks():
    return load_and_chunk_documents()


def test_produces_chunks(chunks):
    assert len(chunks) > 0


def test_every_chunk_has_the_fields_the_pipeline_relies_on(chunks):
    for chunk in chunks:
        assert set(chunk) == {"id", "text", "source", "doc_type"}
        assert chunk["text"].strip()


def test_chunk_ids_are_unique(chunks):
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_doc_type_comes_from_the_folder_name(chunks):
    folders = {d for d in os.listdir(DOCS_FOLDER) if os.path.isdir(os.path.join(DOCS_FOLDER, d))}
    assert {c["doc_type"] for c in chunks} <= folders


@pytest.mark.parametrize("size,overlap", [(200, 40), (400, 80), (800, 160)])
def test_no_chunk_exceeds_the_size_limit(size, overlap):
    for chunk in load_and_chunk_documents(chunk_size=size, chunk_overlap=overlap):
        assert len(chunk["text"]) <= size


def test_smaller_chunk_size_produces_more_chunks():
    small = load_and_chunk_documents(chunk_size=200, chunk_overlap=40)
    large = load_and_chunk_documents(chunk_size=800, chunk_overlap=160)
    assert len(small) > len(large)


def test_every_source_file_is_represented(chunks):
    files = {f for _, _, names in os.walk(DOCS_FOLDER) for f in names if f.endswith(".md")}
    assert {c["source"] for c in chunks} == files


def _chunk_pairs_sharing_boundary_text(chunks):
    """Count consecutive same-source chunks where the end of one reappears in the next."""
    shared = 0
    for first, second in zip(chunks, chunks[1:]):
        if first["source"] != second["source"]:
            continue
        last_line = first["text"].splitlines()[-1].strip()
        if last_line and last_line in second["text"]:
            shared += 1
    return shared


def test_overlap_repeats_text_across_chunk_boundaries():
    with_overlap = load_and_chunk_documents(chunk_size=400, chunk_overlap=120)
    without_overlap = load_and_chunk_documents(chunk_size=400, chunk_overlap=0)
    assert _chunk_pairs_sharing_boundary_text(with_overlap) > 0
    assert _chunk_pairs_sharing_boundary_text(without_overlap) == 0
