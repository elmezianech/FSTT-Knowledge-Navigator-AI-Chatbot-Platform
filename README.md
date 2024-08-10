## Overview

This project is a comprehensive chatbot application designed to provide information about the Faculty of Sciences and Technologies of Tangier (FSTT). The chatbot is capable of handling various queries about the faculty, including details about professors, clubs, courses, and other relevant information. The data is sourced from French PDFs scraped from the FSTT website. The application is built using modern web technologies, ensuring a robust and user-friendly experience.

## Data Source

The chatbot's responses are based on data scraped from the FSTT website [(FSTT Website)](https://fstt.ac.ma/Portail2023/). The scraped data includes French corpus about the faculty, ensuring that the information provided is accurate and up-to-date.

## Technologies Used

- Data Scrapping : Scrappy
- Frontend: Angular, Bootstrap
- Backend: FastAPI, Pydantic, Uvicorn
- Authentication : JWT (JSON Web Tokens)
- Database : MongoDb
- LLM : Ollama llama3
- Vector Database : ChromaDb
- Embeddings: FastEmbedEmbeddings
- CORS Middleware: For handling cross-origin requests

## Project Structure

### Frontend (Angular)

- Landing Page: Users can sign up or sign in.
- Chat Interface: Accessible after logging in, where users can interact with the chatbot and view their conversation history.
- Conversation History: Implement a feature for users to view the history of their conversations.

### Backend (Express.js)
- User Authentication: Manages sign-up, sign-in, and JWT-based authentication.
- Session Management: Stores user queries and the chatbot's responses in MongoDB.

### FastAPI
- RAG Implementation: Uses Ollama Llama3, Langchain, and ChromaDB to process user queries and generate responses based on the scraped FSTT data.

## Functionalities
- User Registration and Login: Users can create an account and log in to access the chatbot.
- Chat Interface: Logged-in users can interact with the chatbot, ask questions, and receive answers.
- Conversation History: Users can view the history of their conversations with the chatbot.
- Information Retrieval: The chatbot can fetch and provide information from the scraped FSTT PDFs, including details about professors, courses, clubs, etc.

