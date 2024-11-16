import os
import time
import getpass
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec 

load_dotenv()

#ファイルの読込
file_paths = ["rewf264_mn.pdf","na_vx3500_unlocked.pdf", "r_hwc62t_e_unlocked.pdf", "30TDL6100_web_unlocked.pdf"]

# Pinecone をLangCainのベクターストアで使用する関数定義
def initialize_vectorstore():
     # Pineconeを使って接続
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass("PINECONE_API_KEY")
    
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")

    pc = Pinecone(api_key=pinecone_api_key)

    # Pinecone (langchain_pinecone) を使ってベクトルストアを作成
    index_name ="manuals"

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    vector_store = PineconeVectorStore(
        index = pc.Index(index_name),
        embedding=OpenAIEmbeddings(),
        )
    return vector_store

# PyMUPDFLoaderで読込、RecursiveCharacterTextSplitterで分割、Pineconeに保存
if __name__ == "__main__":
    # ファイルの読込
    loaders = [PyMuPDFLoader(file_path) for file_path in file_paths]
    raw_docs = []
    for loader in loaders:
        raw_docs.extend(loader.load())
    
    #　テキストをチャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    docs = text_splitter.split_documents(raw_docs)

    # Pineconeに保存
    vectorstore = initialize_vectorstore()
    vectorstore.add_documents(docs)