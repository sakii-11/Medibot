from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

#step 1 : Loading raw PDFs
DATA_PATH ="data/" 
def load_pdf_files(data):
  loader = DirectoryLoader(data, glob='*.pdf', loader_cls = PyPDFLoader)
  documents = loader.load()
  return documents 

documents= load_pdf_files(DATA_PATH)
# print("Length:" ,len(documents))

#step 2 : chunking of data 

def create_chunks(extracted_data):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap =50)
  text_chunks = text_splitter.split_documents(extracted_data)
  return text_chunks 

text_chunks = create_chunks(documents)
# print(len(text_chunks))


# step 3 : create vector embeddings 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(text_chunks, embedding_model)

#Step 4 : Store embeddings in FAISS 
DB_FAISS_PATH="vectorstore/db_faiss"
db.save_local(DB_FAISS_PATH)
