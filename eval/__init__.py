"""
eval — evaluation harness for the RAG system.

  golden_set     — hand-written test questions with expected keywords + reference answers
  metrics        — retrieval metrics (MRR, nDCG@k, keyword coverage); pure functions
  retrieval_eval — builds an index per chunking config and scores retrieval quality
  answer_eval    — runs the real pipeline and scores answers with an LLM-as-judge
"""
