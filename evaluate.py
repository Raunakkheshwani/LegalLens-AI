"""
Runs the retrieval + generation pipeline against a hand-written
evaluation set, then scores it with RAGAS metrics.

Run: python evaluate.py
Requires tests/eval_dataset.json to be filled in with real ground truth.
"""

import json 

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import config
from retriever import get_retriever
from graph import build_graph
from langchain_core.messages import HumanMessage



def load_eval_dataset(path:str ="tests/eval_dataset.json"):
    with open(path, "r") as f:
        return json.load(f)
    
def run_pipeline_for_question(app,query:str):
    """
    Runs the graph up to the human_review pause point, auto-approves
    (no manual input during batch evaluation), and returns the final
    answer plus the raw retrieved chunks used.
    """
    thread_config = {"configurable": {"thread_id": f"eval-{hash(query)}"}}

    initial_state = {
        "query": query,
        "retrieved_chunks": [],
        "draft_answer": "",
        "critique": "",
        "needs_revision": False,
        "loop_count": 0,
        "final_answer": "",
        "messages": [HumanMessage(content=query)],
    }

    ##app.invoke(initial_state, config=thread_config)   # runs until paused at human_review
    ##paused_state = app.get_state(thread_config).values
    #graph me hum input le rhey they toh waha human involver idhr pause hua then driectly humney resume kr diya 
    # Auto-approve: resume without any human edit, since this is a batch eval run
    ##result = app.invoke(None, config=thread_config)
    ##return result["final_answer"], paused_state["retrieved_chunks"]

    result = app.invoke(initial_state, config=thread_config)   # runs straight through to END
    return result["final_answer"], result["retrieved_chunks"]

"""
in json file we have the question and its ground truth so field that are required here is first the QUESTION then uska ANSWER kya aaya , voh answer ka context- CONTEXT_LIST, and the GROUND_TRUTH

"""

def build_ragas_dataset(eval_items):
    app= build_graph(interactive=False)

    questions, answers, context_list, ground_truths= [], [], [], []

    for item in eval_items :
        print(f"Running: {item['question']}")
        answer, chunks= run_pipeline_for_question(app,item['question'])

        questions.append(item['question'])
        answers.append(answer)
        context_list.append(chunks)
        ground_truths.append(item['ground_truth'])

        return Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": context_list,
            "ground_truth": ground_truths

        })

def run_evaluation():
    eval_items= load_eval_dataset()
    dataset= build_ragas_dataset(eval_items)

      # Wrap our Groq LLM + local embeddings so RAGAS uses them instead of OpenAI
    ragas_llm = LangchainLLMWrapper(
        ChatGroq(model=config.LLM_MODEL, groq_api_key=config.GROQ_API_KEY, temperature=0)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    )

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n=== RAGAS EVALUATION RESULTS (baseline, before hybrid search) ===")
    print(results)

    # Save results to disk so we can compare against the "after hybrid search" run later
    results_df = results.to_pandas()
    results_df.to_csv("tests/ragas_baseline_results.csv", index=False)
    print("\nSaved detailed per-question scores to tests/ragas_baseline_results.csv")


if __name__ == "__main__":
    run_evaluation()