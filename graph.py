"""
LangGraph state machine for the Legal Document Review Assistant.

Flow: retrieve -> draft -> critique -> (loop back to draft if issues found) -> END
"""

import os
print("DEBUG:", os.getenv("LANGCHAIN_TRACING_V2"), os.getenv("LANGCHAIN_PROJECT"))



from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma


import config 
from retriever import get_retriever

#STATE
class reviewState(TypedDict): 
    query: str
    messages: Annotated[list,add_messages]
    retrieved_chunk : list[str]
    draft_answer: str
    critique: str
    needs_revision: bool
    loop_count: int
    final_answer: str

llm = ChatGroq(model= config.LLM_MODEL, groq_api_key= config.GROQ_API_KEY, temperature= 0.7)

# ---- 2. Define each Node ----
def retrieve_node(state: reviewState) -> dict:
    retriever= get_retriever() # retrieves the similar chunks acc to user imput query
    docs= retriever.invoke(state['query']) # save the retrieved chunks into this docs including the meta data and all
    chunks= [d.page_content for d in docs] # save the page content in the list 
    return {
        "retrieved_chunk": chunks,
        "messages" : [SystemMessage(content= f"Retrieved {len(chunks)} chunks for query : {state["query"]} ")]
    }

def draft_node(state: reviewState) -> dict:
    # so talking about the draft, the draft will obviously be generate by the llm and that we have to store and make a critique on that and for that we have to invoke the llm and for that we have a prompt which we will be sending to the llm with the RETREIVED CHUNK TO STRUCTURE AND PRECISELY RESPOND HOWEVER to generate the response draft 
    context = "\n\n".join(state["retrieved_chunk"])
    prompt = f"""You are a legal document review assistant.
Using ONLY the context below, answer the user's question clearly.
If the answer isn't in the context, say so honestly — do not make anything up.

Context: 
{context}

Question: {state['query']}

Answer:"""
# in this prompt we collected the chunk and send them in the prompt as context for answering in better way
    response= llm.invoke(prompt)
    return{
        "draft_message": response.content,
        "messages" : [AIMessage(content=f"[Draft] {response.content} ")]
    }

def critique_node(state: reviewState) -> dict:
    context = "\n\n".join(state["retrieved_chunk"])
    prompt = f"""You are a strict fact-checker reviewing a draft answer against source text.

Source context:
{context}

Draft answer:
{state['draft_answer']}

Check: Is every claim in the draft answer directly supported by the source context?
Is anything missing or possibly wrong?

Respond in this exact format:
VERDICT: OK or NEEDS_REVISION
FEEDBACK: <your specific feedback, or "none" if OK>"""
    
    response = llm.invoke(prompt)
    needs_revision= "NEEDS_REVISION" in response.content 

    return {
        "critique" : response.content, 
        "needs_revision" : needs_revision,
        "messages" : [SystemMessage(content = f" [Critiqu] {response.content}")]
    }

def revise_node(state: reviewState) -> dict:
    context = "\n\n".join(state["retrieved_chunk"]),
    prompt = f"""You previously gave this draft answer:
{state['draft_answer']}

A fact-checker gave this feedback:
{state['critique']}

Using ONLY this source context:
{context}

Write a corrected, improved answer to: {state['query']}"""
    
    response=llm.invoke(prompt)
    return {
        "draft_answer" : response.content,
        "loop_count" : state["loop_count"] + 1,
        "message" : [AIMessage(content = f"[Revised Draft] {response.content}")]
    }

def human_review_node(state: reviewState) -> dict:
    # By the time this node actually RUNS, the human has already
    # approved or edited state["draft_answer"] outside the graph.
    # This node just logs that a human checkpoint happened.
    return {
        "messages": [SystemMessage(content="[Human review] Answer approved/edited by human before finalizing.")],
    }

def final_node(state: reviewState) -> dict:
    return {
        "final_answer" : state["draft_answer"],
       "messages" : [AIMessage(content = f" [FINAL_ANSWER] {state['draft_answer']}" )]

    }

## conditional routing--- how to direct the flow of the graph and uunder what conditions 
def route_after_critique(state: reviewState) -> str:
        if state['needs_revision'] and state['loop_count'] < config.MAX_CRITIQUE_LOOPS:
            return "revise" # this is the node name linked to node function of revise node
        return "human_review"
    

# ---- 4. Build + compile with checkpointer + interrupt ----
def build_graph(interactive: bool = True):
    graph=StateGraph(reviewState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("draft", draft_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise", revise_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", final_node)

    graph.add_edge(START,"retrieve")
    graph.add_edge("retrieve","draft")
    graph.add_edge("draft","critique")
# critique then revise then critique then revise then critique...loops
#critique if human review satisfy then final if revise then loops on

    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {"revise": "revise", "human_review" : "human_review"}
    )
    graph.add_edge("revise", "human_review")
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END )

    memory= MemorySaver()
    if interactive:
        # Real usage: pause before human_review so a person can approve/edit.
        return graph.compile(checkpointer=memory, interrupt_before=["human_review"])
    else:
        # Batch evaluation: run straight through, no pause, no human needed.
        return graph.compile(checkpointer=memory)

# ---- 5. Run it with real human interaction ----

if __name__== "__main__":
    app= build_graph()

    user_query = input("enter your legal question about the contract")

    thread_config= {"configurable": {"thread_id": "session-1"}}

    initial_state = {
        "query": user_query,
        "messages": [HumanMessage(content=user_query)],
        "retrieved_chunks": [],
        "draft_answer": "",
        "critique": "",
        "needs_revision": False,
        "loop_count": 0,
        "final_answer": ""
    }

 # Runs retrieve -> draft -> critique -> (maybe revise loop) -> PAUSES before human_review
    app.invoke(initial_state, config= thread_config)


    # using interrupt before human review paused the state to that stage 
    # Graph is now paused. Pull out the current draft to show the human.
    paused_state = app.get_state(thread_config).values
    print("\n=== DRAFT ANSWER — AWAITING YOUR APPROVAL ===")
    print(paused_state["draft_answer"])

    decision = input(
        "\nPress Enter to APPROVE as-is, or type a corrected answer to OVERRIDE it: "
    ).strip()

    if decision:
        # Human is overriding — inject their edited answer back into the state
        app.update_state(thread_config, {"draft_answer": decision})

    # Resume execution from exactly where it paused (human_review -> finalize -> END)
    result = app.invoke(None, config=thread_config)

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])

    print("\n=== FULL REASONING TRACE (messages) ===")
    for msg in result["messages"]:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"[{role}] {msg.content}\n")

