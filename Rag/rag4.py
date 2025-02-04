# This file allow the population of a vector store using the following componants :
# cloud sql (pgvector) we will use an instance of pgvector deployed in cloud storage
# unstrcutured : for partitionning the file 
# gcs : the file to be processed will be stored in gcs

from langchain_google_cloud_sql_pg import PostgresEngine
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_cloud_sql_pg import PostgresVectorStore
from unstructured.partition.md import partition_md
from unstructured.staging.base import dict_to_elements
from unstructured.chunking.title import chunk_by_title
import uuid
import asyncio
from google.cloud import storage
from langchain_core.documents import Document

PROJECT_ID = "x-project-00"
REGION = "europe-west1"
INSTANCE = "vector-db"
DATABASE = "X3000_TurboFixer"
USER = "postgres"
PASSWORD = "admin"
TABLE_NAME = "vector_table2"
BUCKET_NAME = "ai_test_bckt"
FILE_PATH = "doc_latest.md"

embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project = PROJECT_ID)

async def init_vector_table(engine):
    try: 
        await engine.ainit_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=768,  # Vector size for VertexAI model(textembedding-gecko@latest)
        )
    except Exception as e:
        if hasattr(e, "orig") and hasattr(e.orig, "args") and "DuplicateTableError" in e.orig.args[0]:
            print("table already exists")
            return
        else:
            raise e


async def add_document_to_vdb(project_id, region, instance, database, user, password, bucket_name, file_path):
    engine = await PostgresEngine.afrom_instance(
        project_id=project_id, region=region, instance=instance, database=database, user=user, password=password
    )

    await init_vector_table(engine)

    store = await PostgresVectorStore.create(  # Use .create() to initialize an async vector store
        engine = engine,
        table_name = TABLE_NAME,
        embedding_service = embeddings,
    )

    documents,ids = get_partition_document(bucket_name, file_path)

    return await store.aadd_documents(documents, ids=ids)

def get_partition_document(bucket_name, file_path):
    # Create a GCS client
    client = storage.Client()

    # Get the bucket and the blob (file)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    with blob.open("r") as f:
        partitioned_elements = partition_md(file=f)

    elements_dict = [el.to_dict() for el in partitioned_elements if el.category != "UncategorizedText"]

    elements = dict_to_elements(elements_dict)
    chunks = chunk_by_title(
        elements,
        combine_text_under_n_chars=800,
        max_characters=1500,
        # overlap=50
    )

    documents = []
    ids = []
    for element in chunks:
        metadata = element.metadata.to_dict()
        del metadata["languages"]
        metadata["source"] = file_path
        documents.append(Document(page_content=element.text, metadata=metadata))
        ids.append(str(uuid.uuid4()))
    
    return (documents,ids)

if __name__ == "__main__":
    a =asyncio.run(add_document_to_vdb(PROJECT_ID, REGION, INSTANCE, DATABASE, USER, PASSWORD, BUCKET_NAME, FILE_PATH))
    