from langchain_google_vertexai import VertexAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_cloud_sql_pg import PostgresVectorStore, PostgresEngine
from langchain_google_vertexai import VertexAIEmbeddings
from asyncpg.exceptions import UndefinedObjectError
from sqlalchemy.exc import ProgrammingError

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

# retriever = get_retrieval_from_vstore()

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
    return 5

    
@tool(parse_docstring=True)
def order_item(item_name:str, supplier_infos: str, quantity:int) -> str:
    """ 
    Order the given item in case it doesn't existe the stock 

    Args: 
        item_name: name of item to be ordered.
        supplier_infos: supplier contact informations.
        quantity: number of element to be ordered.

    Returns: 
        str: recapitulation of the order
    """
    recap = f"{quantity} element of {item_name} has been ordred from {supplier_infos}"
    return recap

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

def get_db(db_path='checkpoints.db'):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    return memory

def init_vector_table(engine, table_name):
    try:
        engine.init_vectorstore_table(
            table_name=table_name,
            # Vector size for VertexAI model(textembedding-gecko@latest)
            vector_size=768,
        )
    except ProgrammingError as e:
        if hasattr(e, "orig") and hasattr(e.orig, "args") and "DuplicateTableError" in e.orig.args[0]:
            return
        else:
            raise e
    except UndefinedObjectError as e:
        raise UndefinedObjectError from e


def get_remote_vdb_instance(project_id, region, instance, database, user, password, table_name, embeddings):
    engine = PostgresEngine.from_instance(
        project_id=project_id, region=region, instance=instance, database=database, user=user, password=password
    )

    init_vector_table(engine, table_name)

    return PostgresVectorStore.create_sync(  # Use .create() to initialize an async vector store
        engine=engine,
        embedding_service=embeddings,
        table_name=table_name
    )
@tool(parse_docstring=True)
def retrieve_troubleshooting_guide(query: str):
    """
    Search and return information about error code that could occures and how to handle them

    Args:
        query: containing the error code to be searched

    Returns:
        str: instructions to be made 
    """
    res = retriever.invoke(query)
    return res