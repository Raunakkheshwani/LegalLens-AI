"""
LangGraph state machine for the Legal Document Review Assistant.

Flow: retrieve -> draft -> critique -> (loop back to draft if issues found) -> END
"""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages


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

llm = ChatGroq(model= config.LLM_MODEL, groq_api_key= config.GROQ_API_KEY, temparture= 0.7)

# ---- 2. Define each Node ----
def retrieve_node(state: reviewState) -> dict:
    retriever= get_retriever # retrieves the similar chunks acc to user imput query
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