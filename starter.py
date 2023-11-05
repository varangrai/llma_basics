import os.path
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import ServiceContext
service_context = ServiceContext.from_defaults(chunk_size=1000, llm = PaLM())

# check if storage already exists
if (not os.path.exists('./storage')):
    # load the documents and create the index
    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # store it for later
    index.storage_context.persist()
else:
    print("using existing context")
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir='./storage')
    index = load_index_from_storage(storage_context)

retriever = index.as_retriever()
nodes = retriever.retrieve("Who is Paul Graham?")
print(nodes)
# # either way we can now query the index
query_engine = index.as_query_engine(service_context = service_context)
response = query_engine.query("Who is Paul Graham?")
print(response)
