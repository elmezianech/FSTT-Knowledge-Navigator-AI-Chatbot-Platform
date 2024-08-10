from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
import os

folder_path = "pdf"  # Folder containing the PDF files
persist_directory = "db"

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

# Load, split, embed, and store all PDFs in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        pdf_loader = PDFPlumberLoader(file_path)  # Create an instance for each file
        documents = pdf_loader.load_and_split()
        
        chunks = text_splitter.split_documents(documents)
        
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=persist_directory
        )
        


