from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from src.RAGRetriver import get_retriever

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def retrival(query: str) -> list[dict]:
    """
    Get information about topics like turing machines, finite automata, and everything
    that theory of computation topic has to offer. Searches the textbook for relevant information.
    Returns a list of dictionaries, each containing:
    - 'content' : the main content retrieved
    - 'metadata' : metadata about the content
    - 'id' : id of the content
    - 'distance' : distance of the query from this content in vectordb
    """
    return get_retriever().retrive_context(query)


tools = [retrival]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
llm_with_tools = llm.bind_tools(tools)


def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def _build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("agent", chatbot_node)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")
    graph_builder.add_edge("agent", END)
    return graph_builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph
