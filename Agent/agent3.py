from langchain_google_vertexai import VertexAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_core.tools import tool
import warnings
from langchain_google_vertexai import ChatVertexAI
from typing import TypedDict, Annotated
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
import json
from langchain_core.messages import ToolMessage
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode

warnings.filterwarnings('ignore')

def get_vector_store():
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project="swo-trabajo-yrakibi")
# See docker command above to launch a postgres instance with pgvector enabled.
    connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
    collection_name = "X3000_TurboFixer_v3"

    vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    # distance_strategy = DistanceStrategy.COSINE,
    # use_jsonb=True,
)
    
    return vector_store

@tool(parse_docstring=True)
def check_stock(item_name:str) -> int:
    """ 
    Check the stock for the given item

    Args:
        item_name (str): The item name

    Returns: 
        int: number of items of the given name in the stock
    """
    print(item_name)
    return 5

@tool(parse_docstring=True)
def notify_technicien(title: str, criticity: int, message: str) -> str: 
    """
    Notify the technicien that something goes wrong and that he has to make some actions.
    
    Args:
        title: Title of the notification to be send
        criticity: Criticity of the action, varrying from 1 to 5, with:
            1 -> low
            2 -> medium
            3 -> high
            4 -> critical
            5 -> extremly critical
        message: the message to be send to the technicien containg context of the notification and instruction of things to do.
    
    Returns:
        str: the state of the notification send

    """
    return "Notification had beed send"

prompt = """
# Your role
You are an AI assistant at machines' maintenance and reparation .
you will receive notifications codes from machines use the following pieces of retrieved context delimited by XML tags to decide what to do.
did everything that you could to help the user 

<retrieved context>
Retrieved Context:
{context}
</retrieved context>
"""

vector_store = get_vector_store()
retriever = vector_store.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_troubleshooting_guide",
    "Search and return information about error code that could occures and how to handle them"
)

llm = ChatVertexAI(model="gemini-1.5-flash-001", temperature=0)
llm_with_tools = llm.bind_tools([check_stock, notify_technicien, retriever_tool])

class State(TypedDict):
    messages: Annotated[list, add_messages]

def generate(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages":response}

def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

def build_graph(check_stock, notify_technicien, State,add_system, generate, BasicToolNode, route_tools):
    tool_node = ToolNode[check_stock, notify_technicien, retriever_tool]
    
    memory = MemorySaver()

    builder = StateGraph(State)

    builder.add_node("generate",generate)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "system")
    builder.add_edge("system", "generate")
    builder.add_edge("tools", "generate")

    builder.add_conditional_edges(
    "generate",
    route_tools,
    {"tools":"tools",END:END}
)

    return builder.compile(checkpointer=memory)

graph = build_graph(check_stock, notify_technicien, State, add_system, generate, BasicToolNode, route_tools)


thread = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages":user_input}, thread,stream_mode="update"):
        event['messages'][-1].pretty_print()
        # for value in event.values():
        #     print(value)


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(user_input)


