---
title: HackRx Insurance RAG API
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
license: mit
---

# 🏥 HackRx Insurance RAG API

An AI-powered insurance document analysis API built for the HackRx hackathon.

## 🚀 API Endpoint

**POST** `/hackrx/run`

## 🔐 Authentication

Bearer Token: `6bbbd39e54cf65cf7384bbf0011dce9d10a4b7a8818d463d01ba6d016d9acdc2`

## 📖 Usage

```bash
curl -X POST "https://soul-abhi-hackrx-insurance-rag.hf.space/hackrx/run" \
  -H "Authorization: Bearer 6bbbd39e54cf65cf7384bbf0011dce9d10a4b7a8818d463d01ba6d016d9acdc2" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/insurance.pdf",
    "questions": ["What is covered?", "What is the premium?"]
  }'
```

## 🎯 Features

- ✅ Fast PDF processing
- ✅ Google Gemini AI integration
- ✅ Structured JSON responses
- ✅ Bearer token authentication
- ✅ Concurrent question processing
- ✅ Response caching for speed

## 📋 Response Format

```json
{
  "answers": [
    {
      "decision": "Covered/Not Covered/Partially Covered",
      "amount": "Coverage amount details",
      "justification": "Brief explanation"
    }
  ]
}
```

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
