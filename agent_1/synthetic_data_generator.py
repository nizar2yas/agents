from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from typing import  TypedDict, List
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import display_markdown

plan_prompt = """You are an expert technical writer and a machine repair specialist. \
    Your task is to generate a high level outline machines.
Write such an outline for the user provided topic. Give an outline of the documentation along with any relevant notes \
or instructions for the sections.\
    If the user provides critique, or sections to be added,  respond with a revised version of your previous attempts."""

draft_prompt = """
You are an expert technical writer and a machine repair specialist. Your task is to generate detailed 15 page documentation for repairing and using X3000 TurboFixer machine.
Generate the documentation for the user's request and the initial outline.
If the user provides critique,or sections to be added,  respond with a revised version of your previous attempts.
"""

memory = MemorySaver()

model = ChatVertexAI(
    model="gemini-1.5-flash-001",
    temperature=0.5,
    max_retries=2,
)  #reduce inference cost

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    plan_critiques: List[str]
    draft_critiques: List[str]


def generate_plan(state: AgentState):
    content =plan_prompt
    # if "plan_critiques" in state and len(state["plan_critiques"])>0:
    #     critiques ="\n\n".join(state["plan_critiques"])
    #     content = content + f"\n\n here is the user's critiques and feedback : {critiques}"
    # messages = [
    #     SystemMessage(content=content),
    #     HumanMessage(content=state["task"])
    #     ]
    # response = model.invoke(messages)
    plan = """## X3000 TurboFixer Documentation Outline\n\n**I. Introduction**\n\n* **1.1 Overview:** Briefly describe the X3000 TurboFixer, its purpose, and its key applications.\n* **1.2 Target Audience:** Specify who this documentation is intended for (e.g., technicians, operators, maintenance personnel).\n\n**II. Troubleshooting**\n\n* **2.1 Common Issues:** List common problems users might encounter with the X3000 TurboFixer.\n    * **Note:**  Provide concise descriptions and potential causes for each issue.\n* **2.2 Troubleshooting Steps:**  Outline a systematic approach to troubleshooting, including:\n    * **Visual Inspection:**  Checking for obvious signs of damage, loose connections, or obstructions.\n    * **Diagnostic Tests:**  Using built-in diagnostics or external tools to identify specific malfunctions.\n    * **Component Checks:**  Testing individual components (motors, sensors, etc.) to isolate the problem.\n* **2.3 Error Codes:**  Provide a comprehensive list of error codes displayed by the X3000 TurboFixer.\n    * **Note:**  For each code, include a clear description of the error, its potential causes, and recommended troubleshooting steps.\n\n**III. Maintenance and Replacement**\n\n* **3.1 Routine Maintenance:**  Describe regular maintenance tasks required for optimal performance.\n    * **Note:**  Include frequency, procedures, and recommended tools/materials.\n* **3.2 Changeable Parts:**  List all replaceable components of the X3000 TurboFixer.\n    * **Note:**  For each part, provide:\n        * Description and function\n        * Recommended replacement intervals or conditions\n        * Instructions for replacement (if applicable)\n* **3.3 Spare Parts:**  Provide information on obtaining spare parts, including:\n    * Authorized suppliers\n    * Part numbers\n    * Ordering procedures\n\n**IV. Appendix**\n\n* **4.1 Technical Specifications:**  Include detailed technical specifications of the X3000 TurboFixer.\n* **4.2 Diagrams:**  Provide relevant diagrams, such as:\n    * Schematic diagram\n    * Component layout\n    * Wiring diagram\n* **4.3 Glossary:**  Define technical terms used in the documentation.\n\n**Note:**\n\n* This outline is a starting point and can be adapted based on the specific features and complexity of the X3000 TurboFixer.\n* Ensure the documentation is clear, concise, and easy to understand for the target audience.\n* Use visuals (diagrams, illustrations) to enhance comprehension.\n* Include contact information for support and troubleshooting assistance. \n"""
    return {"plan": plan}


def draf(state: AgentState):
    system_content = draft_prompt
    if "draft_critiques" in state and len(state["draft_critiques"])>0:
        critiques ="\n\n".join(state["draft_critiques"])
        system_content = system_content + f"\n\n here is the user's critiques and feedback : {critiques}"

    user_content =f"{state['task']} \n\n here is my plan : \n\n {state['plan']}"
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content)
    ]
    response = model.invoke(messages)
    return {"draf": response.content}

def planner_critique(state:AgentState):
    resp = input("Is the plan is good?\n")

    if resp.lower() =="yes":
        critiques =[]
    else :
        critiques = input("please provide your critiques, separated by ',' ?\n").split(',')
    return {"plan_critiques":critiques}

def draft_critique(state:AgentState):
    resp = input("Is the Doc is good?\n")
    if resp.lower() =="yes":
        critiques = []
    else :
        critiques = input("please provide your critiques, separated by ',' ?\n").split(',')
    return {"draft_critiques":critiques}

def is_plan_good(state):
    if len(state["plan_critiques"]>0) :
        return "planner"
    else :
        return "doc_generator"

def is_docs_good(state):
    if len(state["draft_critiques"]>0) :
        return "planner"
    else :
        return "doc_generator"


builder = StateGraph(AgentState)

builder.add_node("planner", generate_plan)
builder.add_node("planner_critique", planner_critique)
builder.add_node("doc_generator", draf)
builder.add_node("draf_critique", draft_critique)

builder.add_edge(START, "planner")
builder.add_edge("planner", "planner_critique")
builder.add_edge("doc_generator", "draf_critique")

builder.add_conditional_edges(
    "planner_critique",
    is_plan_good,
    {"doc_generator":"doc_generator","planner":"planner"}
)
builder.add_conditional_edges(
    "draf_critique",
    is_docs_good,
    {END:END,"doc_generator":"doc_generator"}
)
graph = builder.compile(checkpointer=memory)

thread = {"configurable": {"thread_id": "1"}}

for s in graph.stream({
    'task': "Could you please generate a documentation for X3000 TurboFixer machine?"
}, thread):
    display_markdown(s)