#This code is working, it do everything nedded 

from langchain_google_vertexai import ChatVertexAI
from typing import TypedDict, Annotated
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langchain.tools.retriever import create_retriever_tool
from utils import get_retrieval_from_vstore, check_stock, notify_technicien, get_retriever_tool
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
# from langgraph.prebuilt import tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
import pprint
import sqlite3

db_path = 'checkpoints.db'
conn = sqlite3.connect(db_path, check_same_thread=False)

memory = SqliteSaver(conn)

plan_prompt = """
    Based on the given context develop an execution plan of what need to be done to repair the machine.
    context:
    {context}
    """

system_promp = """
You are an AI assistant at machines' maintenance and reparation.
you will receive notification code send from the machines, investigate the code and do everything needed to handle it.
"""

retriever = get_retrieval_from_vstore()
agent_tools = [check_stock, notify_technicien, get_retriever_tool(retriever)]
model = ChatVertexAI(model="gemini-1.5-flash-002", temperature=0)
llm_with_tools = model.bind_tools(agent_tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    plan: str

def agent(state: State):
    response = llm_with_tools.invoke(state["messages"])
    # docs ="\n\n".join(doc.page_content for doc in response)
    return {"messages":response} 

def run_tasks(state: State):
    resp = llm_with_tools.invoke(state["messages"])
    return {"messages":resp}

def plan_node(state: State):
    if "plan" not in state or not state["plan"]:
        content = plan_prompt.format(context=state["messages"][-1].content)
        resp = llm_with_tools.invoke(content)
        return {"plan":resp}
    else: 
        return

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

def build_agent():
    tool_node = ToolNode(agent_tools)
    builder = StateGraph(State)

    builder.add_node("generate", agent)
    builder.add_node("tools", tool_node)
    # builder.add_node("plan_node", plan_node)

    builder.add_edge(START, "generate")
    builder.add_edge("tools", "generate")
    # builder.add_edge("plan_node", "generate")
    # builder.add_edge("tools","plan_node")
    # builder.add_edge("plan_node","generate")

    builder.add_conditional_edges(
    "generate",
    route_tools,
    {"tools":"tools",END:END}
    )

    # builder.add_conditional_edges(
    # "plan_node",
    # route_tools,
    # {"tools":"tools",END:END}
    # )

    return builder.compile(checkpointer=memory)
graph = build_agent()

thread = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str):
    for output in graph.stream({"messages":[SystemMessage(content=system_promp), HumanMessage(content= user_input)]}, thread, stream_mode="update"):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

stream_graph_updates("error message : E18XP")
# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye!")
#         break

#     stream_graph_updates(user_input)
