import os
import json
import requests
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

# Essential imports only
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Google AI
import google.generativeai as genai

# Vector DB (simplified)
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    print("Vector DB not available, using direct LLM responses")

# PDF processing
try:
    import fitz
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    print("PDF processing not available")

# Configure Google AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAYOSaY0CpMUG5En8UYv6OE2Hq9ZoXZg")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize model
model = genai.GenerativeModel('gemini-1.5-flash')

# FastAPI app
app = FastAPI(
    title="HackRx Insurance RAG API - Optimized",
    version="2.0.0",
    description="Fast deployment version for hackathon submission"
)

# Auth
security = HTTPBearer()
TEAM_TOKEN = "6bbbd39e54cf65cf7384bbf0011dce9d10a4b7a8818d463d01ba6d016d9acdc2"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return True

# Models
class AnswerResponse(BaseModel):
    decision: str = Field(..., description="Coverage decision")
    amount: str = Field(..., description="Coverage amount")
    justification: str = Field(..., description="Brief explanation")

class RunRequest(BaseModel):
    documents: str = Field(..., description="Document URL")
    questions: List[str] = Field(..., description="Questions to answer")

class RunResponse(BaseModel):
    answers: List[AnswerResponse] = Field(..., description="Structured answers")

# Cache for responses
response_cache = {}

def extract_pdf_text(url: str) -> str:
    """Extract text from PDF URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if PDF_PROCESSING_AVAILABLE:
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
            return "PDF content extraction not available in this deployment"
    except Exception as e:
        return f"Error processing document: {str(e)}"

def answer_insurance_question(question: str, document_text: str) -> AnswerResponse:
    """Answer insurance question using Gemini"""
    cache_key = hashlib.md5(f"{question}{document_text[:100]}".encode()).hexdigest()
    
    # Check cache
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    try:
        prompt = f"""
        Based on this insurance document content:
        {document_text[:2000]}...
        
        Answer this question: {question}
        
        Respond in JSON format:
        {{"decision": "Covered/Not Covered/Partially Covered", "amount": "amount details", "justification": "brief explanation"}}
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=300,
            )
        )
        
        response_text = response.text.strip()
        
        # Extract JSON
        if "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_text = response_text[start:end]
            
            try:
                parsed = json.loads(json_text)
                result = AnswerResponse(
                    decision=parsed.get("decision", "Unable to determine"),
                    amount=parsed.get("amount", "Amount not specified"),
                    justification=parsed.get("justification", "Unable to provide justification")
                )
            except json.JSONDecodeError:
                result = AnswerResponse(
                    decision="Processing Error",
                    amount="N/A",
                    justification="Failed to parse AI response"
                )
        else:
            result = AnswerResponse(
                decision="Processing Error",
                amount="N/A", 
                justification="Invalid response format from AI"
            )
        
        # Cache result
        response_cache[cache_key] = result
        return result
        
    except Exception as e:
        return AnswerResponse(
            decision="Error",
            amount="N/A",
            justification=f"Processing error: {str(e)[:50]}"
        )

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submission(request: RunRequest, authorized: bool = Depends(verify_token)):
    """Process document and answer questions - optimized version"""
    
    try:
        # Extract document text
        print(f"Processing document: {request.documents}")
        document_text = extract_pdf_text(request.documents)
        
        if not document_text or len(document_text) < 10:
            raise HTTPException(status_code=400, detail="Failed to extract document content")
        
        # Process questions concurrently
        print(f"Processing {len(request.questions)} questions")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(answer_insurance_question, question, document_text)
                for question in request.questions
            ]
            
            answers = []
            for future in futures:
                try:
                    result = future.result(timeout=25)
                    answers.append(result)
                except Exception as e:
                    answers.append(AnswerResponse(
                        decision="Timeout",
                        amount="N/A",
                        justification="Processing timeout"
                    ))
        
        return RunResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "HackRx Insurance RAG API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
