#WORKS
import vertexai

from langchain_google_vertexai import ChatVertexAI
from IPython.display import display_markdown
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import List, TypedDict, Annotated
from langgraph.graph.message import add_messages

memory = MemorySaver()

vertexai.init(project="swo-trabajo-yrakibi", location="europe-west1")

model = ChatVertexAI(
    model="gemini-1.5-flash-002",
    temperature=0.8,
    max_retries=2,
) 

draft_prompt = """
You are an expert technical writer and a machine repair specialist. Your task is to generate detailed documentation for maintaining and repairing a fictional machine named : X3000 TurboFixer.
the user has the following outling : 
------
## X3000 TurboFixer Documentation Outline\n\n**II. Troubleshooting**\n\n* **2.1 Common Issues:** List common problems users might encounter with the X3000 TurboFixer.\n    * **Note:**  Provide concise descriptions and potential causes for each issue.\n* **2.2 Troubleshooting Steps:**  Outline a systematic approach to troubleshooting, including:\n    * **Diagnostic Tests:**  Using built-in diagnostics or external tools to identify specific malfunctions.\n    * **Component Checks:**  Testing individual components (motors, sensors, etc.) to isolate the problem.\n* **2.3 Error Codes:**  Provide a comprehensive list of error codes displayed by the X3000 TurboFixer.\n    * **Note:**  For each code, include a clear description of the error, its potential causes, and recommended troubleshooting steps.\n\n**III. Maintenance and Replacement**\n\n* **3.1 Routine Maintenance:**  Describe regular maintenance tasks required for optimal performance.\n    * **Note:**  Include frequency, procedures, and recommended tools/materials.\n* **3.2 Changeable Parts:**  List all replaceable components of the X3000 TurboFixer.\n    * **Note:**  For each part, provide:\n        * Description and function\n        * Recommended replacement intervals or conditions\n        * Instructions for replacement (if applicable)\n* **3.3 Spare Parts:**  Provide information on obtaining spare parts, including:\n    * Authorized suppliers\n    * Part numbers\n    * Ordering procedures\n\n**IV. Appendix**\n\n* **4.1 Technical Specifications:**  Include detailed technical specifications of the X3000 TurboFixer.\n **4.3 Glossary:**  Define technical terms used in the documentation.\n\n**Note:**\n\n* This outline is a starting point and can be adapted based on the specific features and complexity of the X3000 TurboFixer.\n* Ensure the documentation is clear, concise, and easy to understand for the target audience.
------

The documentation will be constrcuted section by section, and the user will provide the sections that he want to documente.
Generate a detailed documentation of each section choosen by the user.
don't give template of what need to be done, but do it 

If the user provides critique respond with a revised version of your previous attempts.
"""

class State(TypedDict):
    messages: Annotated[list[str], add_messages]
    critiques: False

def chatbot(state: State):
    response = [model.invoke(state["messages"])]
    return {"messages": response}

def response_critique(state: State):
    resp = input("Is the Doc is good?\n")
    if resp.lower() =="yes":
        return {"critiques":False}
    else :
        critiques = input("please provide your critiques, separated by '|' ?\n")
        critiques = critiques.split('|')
        return {"messages":critiques, "critiques": True}

def is_good(state: State):
    if state["critiques"]:
        return "chatbot"
    else:
        return END



graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("critique", response_critique)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "critique")

graph_builder.add_conditional_edges(
    "critique",
    is_good,
    {END:END, "chatbot":"chatbot"}
)

graph =graph_builder.compile(checkpointer=memory)

thread = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role":"system", "content":draft_prompt},{"role": "user", "content": user_input}]}, thread):
        for value in event.values():
            print(value)


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(user_input)


