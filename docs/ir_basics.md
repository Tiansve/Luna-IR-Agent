# Information Retrieval Basics

## TF-IDF
Term Frequency–Inverse Document Frequency weights a term by how often it
appears in a document, dampened by how common the term is across the corpus.
It is a strong, cheap baseline for lexical retrieval.

## BM25
BM25 is a probabilistic ranking function that improves on TF-IDF with term
saturation (k1) and document-length normalization (b). It is the default
lexical ranker in Lucene/Elasticsearch.

## Dense Retrieval
Encode queries and documents into vectors with a neural encoder, then rank
by cosine similarity. Strong on paraphrase; weaker than BM25 on rare tokens.

## Hybrid Retrieval
Combine BM25 and dense scores (e.g. reciprocal rank fusion). Usually beats
either component alone.

## Reranking
A second-stage cross-encoder scores (query, doc) pairs jointly. Slow but
accurate; typically applied to the top-50 candidates from the first stage.
