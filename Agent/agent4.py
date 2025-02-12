#This code is working, it do everything nedded 

from langchain_google_vertexai import ChatVertexAI
from typing import TypedDict, Annotated
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from utils import *
from langgraph.prebuilt import ToolNode
# from langgraph.prebuilt import tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

plan_prompt = """
    Based on the given context develop an execution plan of what need to be done to repair the machine.
    context:
    {context}
    """

system_promp = """
You are an AI assistant at machines' maintenance and reparation.
you will receive notification code send from the machines, investigate the code and do everything needed to handle it.
give detailed instruction of what need to be done.
"""

retriever = get_retrieval_from_vstore()
agent_tools = [check_stock, notify_technicien, get_retriever_tool(retriever), order_item]
model = ChatVertexAI(model="gemini-1.5-flash-002", temperature=0)
llm_with_tools = model.bind_tools(agent_tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State):
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

def build_agent():
    memory = MemorySaver()

    tool_node = ToolNode(agent_tools)
    builder = StateGraph(State)

    builder.add_node("generate", agent)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "generate")
    builder.add_edge("tools", "generate")

    builder.add_conditional_edges(
    "generate",
    route_tools,
    {"tools":"tools",END:END}
    )

    return builder.compile(checkpointer=memory)
graph = build_agent()

thread = {"configurable": {"thread_id": "2"}}

# user_input = input("User: ")
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages":[SystemMessage(content=system_promp), HumanMessage(content= user_input)]}, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()
            # for key, value in output.items():
            #     pprint.pprint(f"Output from node '{key}':")
            #     pprint.pprint("---")
            #     pprint.pprint(value, indent=2, width=80, depth=None)
            # pprint.pprint("\n---\n")

# stream_graph_updates("error message : E18XP")
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(user_input)
