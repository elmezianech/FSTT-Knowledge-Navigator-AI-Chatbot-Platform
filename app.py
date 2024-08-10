from fastapi import FastAPI, HTTPException
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

origins = [
    "http://localhost:4200",  # Replace with the origin of your frontend application
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#API_URL = "https://api-inference.huggingface.co/models/arichhi/Llama-2-7b-chat-finetune"
#headers = {"Authorization": "Bearer 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"}

# @app.post("/ask_finetuned")
# async def askPostFineTuned(request: QueryRequest):
#     payload = {
#         "inputs": "<s>[INST] " + request.query + " [/INST]",
#         "parameters": {
#             "max_length": 200,
#             "min_length": 40,
#             "length_penalty": 2.0,
#             "num_beams": 4,
#             "early_stopping": True
#         }
#     }

#     try:
#         response = requests.post(API_URL, headers=headers, json=payload)

#         if response.status_code == 200:
#             result = response.json()[0]["generated_text"]
#             return {"answer": result}
#         else:
#             raise HTTPException(status_code=response.status_code, detail=response.text)
#     except Exception as e:
#         print(f"An error occurred in /ask_finetuned: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

persist_directory = "db"

cached_llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """
    <s>[INST] Vous êtes un assistant technique spécialisé dans la recherche de documents. Utilisez les informations spécifiques contenues dans les documents fournis pour répondre directement à la question. [/INST] </s>
    [INST] Question: {input}
           Informations trouvées: {context}
           Réponse:
    [/INST]
    """
)

@app.post("/ask_rag")
async def askPost(request: QueryRequest):
    try:
        print("Post /ask_rag called")
        print(f"query: {request.query}")

        # Initialize the vector store
        print("Initializing vector store...")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        print("Vector store initialized.")

        # Set up retriever
        print("Setting up retriever...")
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 6,
                "score_threshold": 0.4,
            },
        )
        print("Retriever set up.")

        # Set up document chain
        print("Setting up document chain...")
        document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
        print("Document chain set up.")

        # Create retrieval chain
        print("Creating retrieval chain...")
        chain = create_retrieval_chain(retriever, document_chain)
        print("Retrieval chain created.")

        # Invoke chain
        print("Invoking chain...")
        result = chain.invoke({"input": request.query})
        print("Chain invoked.")

        # Process result
        print("Processing result...")
        sources = []
        for doc in result["context"]:
            sources.append(
                {"source": doc.metadata["source"], "page_content": doc.page_content}
            )
        print("Result processed.")

        response_answer = {"answer": result["answer"], "sources": sources}
        return response_answer

    except Exception as e:
        print(f"An error occurred in /ask_rag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/test_ollama")
# async def test_ollama(request: QueryRequest):
#     try:
#         result = cached_llm.query(request.query)
#         return {"answer": result}
#     except Exception as e:
#         print(f"An error occurred in /test_ollama: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
