import os
import json
import requests
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import google.generativeai as genai

# PDF processing
try:
    import fitz
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Configure Google AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAYOSaY0CpMUG5En8UYv6OE2Hq9ZoXZg")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# FastAPI app
app = FastAPI(
    title="HackRx Insurance RAG API - MINIMAL",
    version="1.0.0",
    description="Minimal working version for urgent deployment"
)

# Auth
security = HTTPBearer()
TEAM_TOKEN = "6bbbd39e54cf65cf7384bbf0011dce9d10a4b7a8818d463d01ba6d016d9acdc2"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return True

# Models
class AnswerResponse(BaseModel):
    decision: str = Field(..., description="Coverage decision")
    amount: str = Field(..., description="Coverage amount")
    justification: str = Field(..., description="Brief explanation")

class RunRequest(BaseModel):
    documents: str = Field(..., description="Document URL")
    questions: List[str] = Field(..., description="Questions")

class RunResponse(BaseModel):
    answers: List[AnswerResponse] = Field(..., description="Answers")

def extract_text_from_url(url: str) -> str:
    """Extract text from document URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if PDF_AVAILABLE and url.lower().endswith('.pdf'):
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            
            text = ""
            doc = fitz.open("temp.pdf")
            for page in doc:
                text += page.get_text()
            doc.close()
            os.remove("temp.pdf")
            return text
        else:
            # Try to get text content directly
            return response.text
    except Exception as e:
        return f"Document processing error: {str(e)}"

def answer_question(question: str, document_text: str) -> AnswerResponse:
    """Answer question using Gemini AI"""
    try:
        prompt = f"""
        You are an insurance expert. Based on this document:
        
        {document_text[:3000]}
        
        Answer this question: {question}
        
        Respond ONLY in this JSON format:
        {{"decision": "Covered/Not Covered/Partially Covered", "amount": "amount or coverage details", "justification": "brief 1-2 line explanation"}}
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=200,
            )
        )
        
        text = response.text.strip()
        
        # Extract JSON
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
            
            try:
                data = json.loads(json_str)
                return AnswerResponse(
                    decision=data.get("decision", "Unable to determine"),
                    amount=data.get("amount", "Not specified"),
                    justification=data.get("justification", "Processing completed")
                )
            except:
                pass
        
        # Fallback response
        return AnswerResponse(
            decision="Processed",
            amount="See document for details",
            justification="AI analysis completed but format parsing failed"
        )
        
    except Exception as e:
        return AnswerResponse(
            decision="Error",
            amount="N/A",
            justification=f"Processing error: {str(e)[:100]}"
        )

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submission(request: RunRequest, authorized: bool = Depends(verify_token)):
    """Main API endpoint"""
    try:
        # Get document content
        document_text = extract_text_from_url(request.documents)
        
        if len(document_text) < 10:
            raise HTTPException(status_code=400, detail="Failed to extract document content")
        
        # Process questions
        answers = []
        for question in request.questions:
            answer = answer_question(question, document_text)
            answers.append(answer)
        
        return RunResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "HackRx Insurance API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy", "api_key_configured": bool(GOOGLE_API_KEY)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
