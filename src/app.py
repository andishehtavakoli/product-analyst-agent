import os
import openai
import chainlit as cl

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core import SQLDatabase
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)

engine = create_engine("postgresql://postgres:mypass@localhost:5432/postgres")
metadata_obj = MetaData()
sql_database = SQLDatabase(engine, include_tables=["data_stats"])

llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
except:
    # file_path = '/Users/andishehtavakoli/Documents/github-project/product-analyst-agent/data/sample.txt'
    # documents = SimpleDirectoryReader(input_files=[file_path]).load_data(show_progress=True)
    # index = VectorStoreIndex.from_documents(documents)
    # index.storage_context.persist()
    # set Logging to DEBUG for more detailed outputs
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        (SQLTableSchema(table_name="data_stats"))
    ]  # add a SQLTableSchema for each table

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,

    )
    
    obj_index.persist(persist_dir="./storage")
    query_engine = SQLTableRetrieverQueryEngine(sql_database, obj_index.as_retriever(similarity_top_k=1),)
    
 


@cl.on_chat_start
async def start():
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.context_window = 4096
    
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        (SQLTableSchema(table_name="data_stats"))
    ]  # add a SQLTableSchema for each table
    # service_context = ServiceContext.from_defaults(callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))
    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,

    )
    
    obj_index.persist(persist_dir="./storage")
    query_engine = SQLTableRetrieverQueryEngine(sql_database, obj_index.as_retriever(similarity_top_k=1),)
    
    cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


# @cl.on_message
# async def main(message: cl.Message):
#     query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine

#     msg = cl.Message(content="", author="Assistant")

#     res = await cl.make_async(query_engine.query)(message.content)


#     for token in res.response:
#         await msg.stream_token(token)
#     await msg.send()
    
import chainlit as cl


@cl.on_message
async def on_message(msg: cl.Message):
    if cl.context.session.client_type == "copilot":
        fn = cl.CopilotFunction(name="test", args={"msg": msg.content})
        res = await fn.acall()
        query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine
        res = await cl.make_async(query_engine.query)(msg.content)
        
        msg = cl.Message(content=res, author="Assistant")


        for token in res.response:
            await msg.stream_token(token)
        await msg.send()
   
        
        
 

