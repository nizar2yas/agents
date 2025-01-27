from langchain_google_vertexai import VertexAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_core.tools import tool
import warnings
from langchain_google_vertexai import ChatVertexAI
from typing import List, TypedDict, Annotated
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
import json
from langchain_core.messages import ToolMessage

warnings.filterwarnings('ignore')
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

vector_store = get_vector_store()

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

llm = ChatVertexAI(model="gemini-1.5-flash-001", temperature=0)
llm_with_tools = llm.bind_tools([check_stock, notify_technicien])

class State(TypedDict):
    question: str
    context: List[Document]
    messages: Annotated[list, add_messages]

def retriever(state: State):
    retrieved_documents = vector_store.similarity_search(state["question"])
    return {"context":retrieved_documents}

def generate(state: State):
    docs ="\n\n".join(doc.page_content for doc in state["context"])
    messages =[
        SystemMessage(content=prompt.format(context=docs)),
        HumanMessage(content=state["question"])
    ]
    response = llm_with_tools.invoke(messages)
    return {"messages":response}

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

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

def build_graph(check_stock, notify_technicien, State, retriever, generate, BasicToolNode, route_tools):
    tool_node = BasicToolNode(tools=[check_stock, notify_technicien])
    memory = MemorySaver()

    builder = StateGraph(State)

    builder.add_node("retriever",retriever)
    builder.add_node("generate",generate)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "generate")
    builder.add_edge("tools", "generate")

    builder.add_conditional_edges(
    "generate",
    route_tools,
    {"tools":"tools",END:END}
)

    graph =builder.compile(checkpointer=memory)

    graph = builder.compile()
    return graph

graph = build_graph(check_stock, notify_technicien, State, retriever, generate, BasicToolNode, route_tools)


thread = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream({"question":user_input}, thread,stream_mode="update"):
        # event['messages'][-1].pretty_print()
        for value in event.values():
            print(value)


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(user_input)


