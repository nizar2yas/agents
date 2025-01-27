import warnings
# import unstructured_pytesseract
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from typing import List, TypedDict
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage


warnings.filterwarnings('ignore')

llm = ChatVertexAI(model="gemini-1.5-flash-001", temperature=0)

embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project="swo-trabajo-yrakibi")
# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "X3000_TurboFixer"


vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    # distance_strategy = DistanceStrategy.COSINE,
    # use_jsonb=True,
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


prompt_template = """
# Your role
You are an expert at maintenance and reparation of machines, users will ask you informations about machines, or how to resolve issue that they face.


# Instruction
Your task is to answer the question and give instruction if needed, using the following pieces of retrieved context delimited by XML tags.

<retrieved context>
Retrieved Context:
{context}
</retrieved context>


# Constraint
1. Choose the most relevant content(the key content that directly relates to the question) from the retrieved context and use it to generate an answer.
2. Generate a concise, logical answer. When generating the answer, Do Not just list your selections, But rearrange them in context so that they become paragraphs with a natural flow. 
3. When you don't have retrieved context for the question or If you have a retrieved documents, but their content is irrelevant to the question, you should answer 'I can't find the answer to that question in the material I have'.
"""
prompt = ChatPromptTemplate([(prompt_template)])


def retriever(state: State):
    retrieved_documents = vector_store.similarity_search(state["question"])
    return {"context":retrieved_documents}


def generate(state: State):
    docs ="\n\n".join(doc.page_content for doc in state["context"])
    messages =[
        SystemMessage(content=prompt_template.format(context=docs)),
        HumanMessage(content=state["question"])
    ]
    response = llm.invoke(messages)
    return {"answer":response.content}


memory = MemorySaver()

builder =StateGraph(State)

builder.add_node("retriever",retriever)
builder.add_node("generate",generate)

builder.add_edge(START, "retriever")
builder.add_edge("retriever", "generate")
builder.add_edge("generate", END)

graph =builder.compile(checkpointer=memory)

graph = builder.compile()

thread = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    # as seen in the course, I could use 'value' or 'update': 
    # for event in graph.stream({"question":user_input}, thread, stream_mode="updates"):
    for event in graph.stream({"question":user_input}, thread):
        for value in event.values():
            value.pretty_print()
        print("---"*25)


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(user_input)


