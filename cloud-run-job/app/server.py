from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_google_cloud_sql_pg import PostgresVectorStore, PostgresEngine
import os
from langchain_google_vertexai import VertexAIEmbeddings
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

PROJECT_ID = os.getenv("PROJECT_ID", "x-project-00")
REGION = os.getenv("REGION", "europe-west1")
INSTANCE = os.getenv("INSTANCE", "vector-db")
DATABASE = os.getenv("DATABASE", "X3000_TurboFixer")
# USER = os.getenv("USER", "postgres")
USER =  "postgres"
TABLE_NAME = os.getenv("TABLE_NAME", "vector_table2")
INPUT_BUCKET_NAME = os.getenv("INTPUT_BUCKET_NAME", "rag_input")
OUTPUT_BUCKET_NAME = os.getenv("OUTPUT_BUCKET_NAME", "rag_output_files")
FILE_PATH = os.getenv("FILE_PATH")
SECRET_ID = os.getenv("SECRET_ID", "vector_db_secret")
PASSWORD = os.getenv("password")

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# (1) Initialize VectorStore
def init_vector_table(engine):
    try:
        logger.info("Initializing vector table...")
        engine.init_vectorstore_table(
            table_name=TABLE_NAME,
            # Vector size for VertexAI model(textembedding-gecko@latest)
            vector_size=768,
        )
        logger.info("Vector table initialized successfully.")
    except Exception as e:
        if hasattr(e, "orig") and hasattr(e.orig, "args") and "DuplicateTableError" in e.orig.args[0]:
            logger.warning("Table already exists.")
            return
        else:
            logger.error("Error initializing vector table: %s", e)
            raise e
        
def instantiate_db():
    embeddings = VertexAIEmbeddings( model_name="text-embedding-004", project=PROJECT_ID)

    engine = PostgresEngine.from_instance(
        project_id=PROJECT_ID, region=REGION, instance=INSTANCE, database=DATABASE, user=USER, password=PASSWORD
    )

    init_vector_table(engine)

    return PostgresVectorStore.create_sync(  # Use .create() to initialize an async vector store
        engine=engine,
        embedding_service=embeddings,
    )

vector_store = instantiate_db()

# (2) Build retriever
def concatenate_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
notes_retriever = vector_store.as_retriever() | concatenate_docs


# (3) Create prompt template
prompt_template = PromptTemplate.from_template(
    """You are a Cloud Run expert answering questions. 
Use the retrieved release notes to answer questions
Give a detailed answer, and if you are unsure of the answer, just say so.

Release notes: {notes}

Here is your question: {query}
Your answer: """)

# (4) Initialize LLM
llm = ChatVertexAI(model_name="gemini-1.5-flash-002", temperature=0, top_k=3)

# (5) Chain everything together
chain = (
    RunnableParallel({
        "notes": notes_retriever,
        "query": RunnablePassthrough()
    })
    | prompt_template
    | llm
    | StrOutputParser()
)


add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
