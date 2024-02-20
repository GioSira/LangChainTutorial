import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document


# setup langsmith
load_dotenv()
os.getenv('LANGCHAIN_TRACING_V2')
os.getenv('LANGCHAIN_API_KEY')

# run ollama on shell
# ollama run phi

# fetch the model
llm = Ollama(model=f'phi')

# load the documents in the following website
# the documents will be inserted into a vector-store, i.e. word embedding db
loader = WebBaseLoader("https://docs.smith.langchain.com")
docs = loader.load()

# load embeddings for the vector-store
# default is LLAMA2
embeddings = OllamaEmbeddings(model='phi')

# split document and store into vector-db
# FAISS is for Facebook AI Similarity Search
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# define prompt for RAG
template = """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}

Answer:
"""

# create the prompt template
prompt = ChatPromptTemplate.from_template(template)

# simple chain for RAG
document_chain = create_stuff_documents_chain(llm, prompt)

# ========================================= TEST THE MODEL ====================================

# test the model on a single input-context pair
# res = document_chain.invoke({
#    "input": "how can langsmith help with testing?",
#    "context": [Document(page_content="langsmith can let you visualize test results")]
# })

# print('Testing\n', res)

# ===============================================================================================

# get automatically the context according to query

# enable similarity search
retriever = vector.as_retriever()
# retrieval chain automatically substitutes context with the retrieved content
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# query the model
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})

print(response['answer'])
