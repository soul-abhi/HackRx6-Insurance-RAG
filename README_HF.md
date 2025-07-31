---
title: HackRx Insurance RAG API
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.7.1
app_file: app_hf.py
pinned: false
license: mit
---

# HackRx Insurance RAG API

Fast deployment version for hackathon submission.

## API Endpoint

`POST /hackrx/run`

## Authentication

Bearer Token: `6bbbd39e54cf65cf7384bbf0011dce9d10a4b7a8818d463d01ba6d016d9acdc2`

## Usage

```bash
curl -X POST "https://your-space.hf.space/hackrx/run" \
  -H "Authorization: Bearer 6bbbd39e54cf65cf7384bbf0011dce9d10a4b7a8818d463d01ba6d016d9acdc2" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/insurance.pdf",
    "questions": ["What is covered?", "What is the premium?"]
  }'
```
