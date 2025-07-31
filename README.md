# HackRx6 Insurance RAG System

A high-performance Retrieval-Augmented Generation (RAG) system for insurance document processing and question answering.

## 🚀 Features

- **FastAPI** web framework with Bearer token authentication
- **Google Gemini LLM** for intelligent responses
- **ChromaDB** vector database for document storage
- **Concurrent processing** for multiple questions
- **Response caching** for improved performance
- **Sub-10 second response times** for hackathon requirements

## 📋 API Endpoints

### POST `/hackrx/run`

Process insurance documents and answer questions.

**Headers:**

```
Authorization: Bearer 6bbbd39e54cf65cf7384bbf0011dce9d10a4b7a8818d463d01ba6d016d9acdc2
Content-Type: application/json
```

**Request Body:**

```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is covered under this policy?",
    "What are the premium amounts?"
  ]
}
```

**Response:**

```json
{
  "answers": [
    {
      "decision": "Covered",
      "amount": "$50,000",
      "justification": "Medical expenses are covered up to policy limit"
    }
  ]
}
```

## 🛠 Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `config/.env`
4. Run: `python api_server.py`

## 🌐 Deployment

This project is configured for deployment on:

- Render
- Railway
- Heroku
- AWS/GCP/Azure

## 📊 Performance

- **2 questions**: ~6-8 seconds
- **5+ questions**: ~8-12 seconds
- **Cached responses**: <2 seconds

## 🏆 HackRx6 Competition

Optimized for hackathon submission with professional code quality and highest accuracy requirements.
