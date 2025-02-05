import os
import logging
from langchain_google_cloud_sql_pg import PostgresEngine
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_cloud_sql_pg import PostgresVectorStore
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import dict_to_elements
from unstructured.chunking.title import chunk_by_title
import uuid
import asyncio
from google.cloud import storage
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_ID = os.getenv("PROJECT_ID", "x-project-00")
REGION = os.getenv("REGION", "europe-west1")
INSTANCE = os.getenv("INSTANCE", "vector-db")
DATABASE = os.getenv("DATABASE", "X3000_TurboFixer")
USER = os.getenv("USER", "postgres")
TABLE_NAME = os.getenv("TABLE_NAME", "vector_table2")
INPUT_BUCKET_NAME = os.getenv("INTPUT_BUCKET_NAME", "rag_input")
OUTPUT_BUCKET_NAME = os.getenv("OUTPUT_BUCKET_NAME", "rag_output_files")
FILE_PATH = os.getenv("FILE_PATH")
SECRET_ID = os.getenv("SECRET_ID", "vector_db_secret")
PASSWORD = os.getenv("password","admin")

embeddings = VertexAIEmbeddings(
    model_name="text-embedding-004", project=PROJECT_ID)


async def init_vector_table(engine):
    try:
        logger.info("Initializing vector table...")
        await engine.ainit_vectorstore_table(
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


async def add_document_to_vdb(project_id, region, instance, database, user, password, bucket_name, file_path, client):
    logger.info("Connecting to Postgres instance...")
    engine = await PostgresEngine.afrom_instance(
        project_id=project_id, region=region, instance=instance, database=database, user=user, password=password
    )
    logger.info("Connected to Postgres instance.")

    await init_vector_table(engine)

    store = await PostgresVectorStore.create(  # Use .create() to initialize an async vector store
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=embeddings,
    )
    logger.info("Vector store created.")

    documents, ids = get_partition_document(client, bucket_name, file_path)
    logger.info("Documents partitioned and prepared for insertion.")

    result = await store.aadd_documents(documents, ids=ids)
    logger.info("Documents added to vector store.")
    return result


def get_partition_document(client, bucket_name, file_path):
    logger.info("Fetching file from GCS bucket...")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    _, extension = os.path.splitext(file_path)
    with blob.open("r") as f:
        match extension:
            case ".pdf":
                partitioned_elements = partition_pdf(file=f)
            case ".md":
                partitioned_elements = partition_md(file=f)
            case _:
                raise Exception(f"insupported type {
                                extension}, only 'pdf', and 'md' are supported")
    logger.info("File partitioned.")

    elements_dict = [
        el.to_dict() for el in partitioned_elements if el.category != "UncategorizedText"]
    elements = dict_to_elements(elements_dict)
    chunks = chunk_by_title(
        elements,
        combine_text_under_n_chars=800,
        max_characters=1500,
        # overlap=50
    )
    logger.info("Elements chunked.")

    documents = []
    ids = []
    for element in chunks:
        metadata = element.metadata.to_dict()
        del metadata["languages"]
        metadata["source"] = file_path
        documents.append(
            Document(page_content=element.text, metadata=metadata))
        ids.append(str(uuid.uuid4()))

    logger.info("Documents and IDs prepared.")
    return (documents, ids)


def move_blob(storage_client, bucket_name, blob_name, destination_bucket):

    logger.info(f"moving file {blob_name} from {bucket_name} to {destination_bucket}")

    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.bucket(destination_bucket)

    _ = source_bucket.copy_blob(
        source_blob, destination_bucket
    )
    source_bucket.delete_blob(blob_name)

    logger.info(f"file has been moved successfully")


if __name__ == "__main__":
    logger.info("Starting the process to add document to vector database...")
    logger.debug(f"the following variables are used : \n PROJECT_ID: {PROJECT_ID}\nREGION: {REGION}\nINSTANCE: {INSTANCE}\nDATABASE: {DATABASE}\nUSER: {USER}\nTABLE_NAME: {
                 TABLE_NAME}\nINPUT_BUCKET_NAME: {INPUT_BUCKET_NAME}\nFILE_PATH: {FILE_PATH}\nSECRET_ID = {SECRET_ID}\nOUTPUT_BUCKET_NAME = {OUTPUT_BUCKET_NAME}")

    storage_client = storage.Client()
    asyncio.run(add_document_to_vdb(PROJECT_ID, REGION, INSTANCE, DATABASE,
                USER, PASSWORD, INPUT_BUCKET_NAME, FILE_PATH, storage_client))

    move_blob(storage_client, INPUT_BUCKET_NAME, FILE_PATH, OUTPUT_BUCKET_NAME)
    logger.info("Process completed.")
