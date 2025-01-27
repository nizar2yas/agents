from langchain_google_vertexai import VertexAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool

def get_vector_store(collection_name="X3000_TurboFixer_v3"):
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project="swo-trabajo-yrakibi")
# See docker command above to launch a postgres instance with pgvector enabled.
    connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!

    return PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    # distance_strategy = DistanceStrategy.COSINE,
    # use_jsonb=True,
    )

def get_retrieval_from_vstore(collection_name="X3000_TurboFixer_v3"):
    vector_store = get_vector_store(collection_name)
    return vector_store.as_retriever()

def get_retriever_tool(retriever):
    return create_retriever_tool(
        retriever,
        "retrieve_troubleshooting_guide",
        "Search and return information about error code that could occures and how to handle them"
    )
    
@tool(parse_docstring=True)
def check_stock(item_name:str) -> int:
    """ 
    Check the stock for the given item

    Args:
        item_name: The item name*

    Returns: 
        int: number of items of the given name in the stock
    """
    print(item_name)
    return 0

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
