"""
Runs the retrieval + generation pipeline against a hand-written
evaluation set, then scores it with RAGAS metrics.

Run: python evaluate.py
Requires tests/eval_dataset.json to be filled in with real ground truth.
"""

import json 

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import _faithfulness, _answer_relevance, _context_precision, _context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import config
from retriever import get_retriever
from graph import build_graph
from langchain_core.messages import HumanMessage

