import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

#Step 1 : Setup LLM (Mistral with Huggingface)
HF_TOKEN = os.getenv("HF_TOKEN")
huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
  llm = HuggingFaceEndpoint(
    repo_id=huggingface_repo_id,
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.5,
    model_kwargs={
      "max_length":"512"
    }
  )
  return llm

#step 2 : Connect LLM with FAISS and create chain 

CUSTOM_PROMPT_TEMPLATE = """
You are an AI-powered medical assistant. Use ONLY the information provided in the context to answer the user's question accurately and concisely.  

- If the answer is not found in the context.
- Do NOT make up answers or provide opinions.  
- Do NOT generate responses that are outside the provided context.  

Context: {context}  
Question: {question}  

Start the answer directly. No small talk please. Provide a clear, direct, and factual response based strictly on the given context.
"""
def set_custom_prompt(custom_prompt_template):
  prompt = PromptTemplate(template=custom_prompt_template, input_vairables=["context","question"])
  return prompt

#Load Database 
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True) #it is true since we trust the source (pdf) completely

#Create QA Chain 
qa_chain = RetrievalQA.from_chain_type(
  llm= load_llm(huggingface_repo_id),
  chain_type="stuff",
  retriever=db.as_retriever(search_kwargs={'k':3}),
  return_source_documents=True,
  chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

#Invoke with a single query
user_query = input("Write Query Here: ")
response=qa_chain.invoke({'query':user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])