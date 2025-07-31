import os
import json
import requests
import hashlib
import urllib.parse
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from typing import Any, List, Optional, Iterator, AsyncIterator
from llama_index.core.base.llms.types import ChatResponse, LLMMetadata, CompletionResponse, CompletionResponseGen, \
    ChatResponseGen
from llama_index.core.llms import LLM, ChatMessage, MessageRole

from llama_index.core.output_parsers import PydanticOutputParser  # Ensure this import is present
from llama_index.core.program import LLMTextCompletionProgram  # Ensure this import is present

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from dotenv import load_dotenv

import google.generativeai as genai
import fitz

# --- 0. Load Environment Variables ---
# Try multiple paths for .env file
import os
if os.path.exists('config/.env'):
    load_dotenv('config/.env')
elif os.path.exists('.env'):
    load_dotenv('.env')
else:
    # For production, environment variables should be set directly
    pass

# --- ChromaDB Configuration (Global) ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# --- Google Gemini LLM Configuration (Global) ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class CustomGeminiLLM(LLM):
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.3  # Lower temperature for faster, more consistent responses
    max_tokens: Optional[int] = 512  # Limit tokens for faster responses
    api_key: str = os.getenv("GOOGLE_API_KEY")

    _client: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.api_key is None:
            raise ValueError("GOOGLE_API_KEY must be set in your environment.")
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self.model_name)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _get_text_from_content(self, content: Any) -> str:
        if content and hasattr(content, 'parts') and content.parts:
            for part in content.parts:
                if hasattr(part, 'text'):
                    return part.text
        return ""

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if "formatted" in kwargs:
            kwargs.pop("formatted")
        response = self._client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
            **kwargs,
        )
        text = self._get_text_from_content(response.candidates[0].content)
        return CompletionResponse(text=text, raw=response)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        if "formatted" in kwargs:
            kwargs.pop("formatted")
        response_gen = self._client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
            stream=True,
            **kwargs,
        )
        for chunk in response_gen:
            text = self._get_text_from_content(chunk.candidates[0].content)
            yield CompletionResponse(text=text, raw=chunk)

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncIterator[CompletionResponseGen]:
        for response in self.stream_complete(prompt, **kwargs):
            yield response

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        if "formatted" in kwargs:
            kwargs.pop("formatted")
        gemini_messages = []
        for message in messages:
            role = "user" if message.role == MessageRole.USER else "model"
            gemini_messages.append({"role": role, "parts": [{"text": message.content}]})

        response = self._client.generate_content(
            gemini_messages,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
            **kwargs,
        )
        text = self._get_text_from_content(response.candidates[0].content)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), raw=response)

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        if "formatted" in kwargs:
            kwargs.pop("formatted")
        gemini_messages = []
        for message in messages:
            role = "user" if message.role == MessageRole.USER else "model"
            gemini_messages.append({"role": role, "parts": [{"text": message.content}]})

        response_gen = self._client.generate_content(
            gemini_messages,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
            stream=True,
            **kwargs,
        )
        for chunk in response_gen:
            text = self._get_text_from_content(chunk.candidates[0].content)
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), raw=chunk)

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> AsyncIterator[ChatResponseGen]:
        for response in self.stream_chat(messages, **kwargs):
            yield response


# --- Global LLM and Embedding Model Initialization ---
# This runs once when the server starts.
gemini_llm_model = CustomGeminiLLM(model_name="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Using faster, smaller embedding model for speed
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
Settings.llm = gemini_llm_model  # Set the global LLM for LlamaIndex to use by default

# Initialize response cache
import time
from functools import lru_cache
response_cache = {}
cache_ttl = 3600  # 1 hour cache

print("\n--- Core LLM and Embedding Models Initialized for API Server ---")


# --- Dynamic Document Fetching and Processing Function ---
def fetch_and_process_document_from_url(document_url: str):
    """
    Fetches a document from a URL, extracts text, and returns a LlamaIndex Document.
    Currently primarily supports PDF files.
    """
    documents = []
    print(f"\n--- Fetching and processing document from URL: {document_url} ---")
    try:
        # Correctly determine file type based on URL path, ignoring query parameters
        parsed_url = urllib.parse.urlparse(document_url)
        path = parsed_url.path
        file_extension = os.path.splitext(path)[1].lower().lstrip('.')

        # Download the document
        response = requests.get(document_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # Process based on file type
        if file_extension == 'pdf':
            temp_file_path = "temp_document.pdf"
            with open(temp_file_path, "wb") as f:
                f.write(response.content)

            text = ""
            doc = fitz.open(temp_file_path)
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            os.remove(temp_file_path)  # Clean up temporary file

            documents.append(Document(
                text=text,
                metadata={"file_name": os.path.basename(path), "file_path": document_url, "file_type": "pdf"}
            ))
            print(f" - Successfully extracted text from PDF: {document_url}")

        else:
            print(f" - Unsupported file type '{file_extension}' from URL: {document_url}. Skipping.")

    except requests.exceptions.RequestException as e:
        print(f" - Error downloading document from {document_url}: {e}")
    except Exception as e:
        print(f" - Error processing document {document_url}: {e}")

    if not documents:
        print(f"--- Document Processing Failed for URL: {document_url}. No documents loaded. ---")
    else:
        print(
            f"--- Document Processing Complete for URL: {document_url}. Extracted text from {len(documents)} document(s). ---")
    return documents


# --- Dynamic Vector Store Index Setup Function (Modified for Namespaces) ---
# This function will now be called per API request within the endpoint
def setup_vector_store_index_for_request(documents: List[Document], collection_name: str, namespace: str):
    """
    Sets up or connects to a ChromaDB collection within a specific namespace and populates it with documents.
    Returns the LlamaIndex VectorStoreIndex for that namespace.
    """
    print(f"\n--- Setting up Vector Store Index for namespace '{namespace}' in collection '{collection_name}'... ---")

    # Create a unique collection name using namespace
    unique_collection_name = f"{collection_name}_{namespace}"
    
    # Get or create the ChromaDB collection
    chroma_collection = chroma_client.get_or_create_collection(name=unique_collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Check if the collection already contains vectors
    collection_count = chroma_collection.count()

    if collection_count == 0:  # Only add if collection is empty
        print(f"--- Collection '{unique_collection_name}' is empty. Parsing and generating embeddings for new documents... ---")
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)  # Smaller chunks for speed
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)  # Disable progress for speed

        # Batch embedding generation for better performance
        if Settings.embed_model:
            print(f"--- Generating embeddings for {len(nodes)} nodes in batch... ---")
            texts = [node.get_content() for node in nodes]
            embeddings = Settings.embed_model.get_text_embedding_batch(texts)
            
            for i, node in enumerate(nodes):
                node.embedding = embeddings[i]
        else:
            raise ValueError("Settings.embed_model is not set. Cannot generate embeddings for nodes.")
        print(f"--- Embeddings generated for {len(nodes)} nodes. ---")

        print(f"--- Adding {len(nodes)} nodes with embeddings to ChromaDB collection '{unique_collection_name}'... ---")
        vector_store.add(nodes)
        print(f"--- Nodes added to ChromaDB collection '{unique_collection_name}'. ---")

        # --- Index creation for empty collection ---
        index = VectorStoreIndex(
            nodes=nodes,  # Pass the nodes which already have embeddings
            vector_store=vector_store,  # Pass the configured ChromaDB vector store
            embed_model=Settings.embed_model  # Explicitly ensure HuggingFaceEmbedding is used
        )
    else:  # Collection already contains vectors
        print(
            f"--- Collection '{unique_collection_name}' already contains {collection_count} vectors. Skipping adding nodes. ---")
        # --- Index creation for non-empty collection ---
        # Create the LlamaIndex VectorStoreIndex from the existing vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=Settings.embed_model  # Pass embed_model here too for consistency
        )

    print(f"--- Vector Store Index for namespace '{namespace}' ready. ---")
    return index


# --- Pydantic Models for Query Details (for LLM Parser) ---
class QueryDetails(BaseModel):
    """
    Extracted structured details from a natural language query.
    """
    age: Optional[int] = Field(None, description="Age of the person, if mentioned in the query.")
    gender: Optional[str] = Field(None, description="Gender of the person (e.g., 'male', 'female'), if mentioned.")
    medical_procedure: Optional[str] = Field(None,
                                             description="Specific medical procedure or condition mentioned (e.g., 'knee surgery', 'dental treatment').")
    location: Optional[str] = Field(None, description="Geographical location mentioned (e.g., 'Pune', 'Mumbai').")
    policy_duration: Optional[str] = Field(None,
                                           description="Duration of the insurance policy (e.g., '3-month policy', '1 year').")


class AnswerResponse(BaseModel):
    decision: str = Field(..., description="The decision result (e.g., 'Covered', 'Not Covered', 'Partially Covered')")
    amount: str = Field(..., description="The coverage amount or relevant financial details")
    justification: str = Field(..., description="One to two line clear explanation of the decision")


# --- Concurrent Query Processing Function ---
def process_question_batch(questions: List[str], index: VectorStoreIndex, llm_model: Any) -> List[AnswerResponse]:
    """
    Process multiple questions concurrently for faster response times.
    """
    with ThreadPoolExecutor(max_workers=3) as executor:  # Limit concurrent requests
        futures = [
            executor.submit(query_insurance_rag, question, index, llm_model) 
            for question in questions
        ]
        
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=30):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in concurrent processing: {e}")
                error_response = AnswerResponse(
                    decision="Error",
                    amount="N/A",
                    justification=f"Processing timeout or error: {str(e)[:50]}"
                )
                results.append(error_response)
    
    return results


# --- FastAPI Application Instance ---
app = FastAPI(
    title="HackRx LLM-Powered Query-Retrieval System API",
    version="1.0.0",
    description="API for processing documents and answering questions contextually."
)

# --- Authentication ---
security = HTTPBearer()
TEAM_TOKEN = "6bbbd39e54cf65cf7384bbf0011dce9d10a4b7a8818d463d01ba6d016d9acdc2"  # From your problem statement


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != TEAM_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


# --- Pydantic Models for Request and Response ---
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL of the document to process (e.g., PDF Blob URL).")
    questions: List[str] = Field(..., description="A list of natural language questions to answer.")


class RunResponse(BaseModel):
    answers: List[AnswerResponse] = Field(..., description="A list of structured answers with decision, amount, and justification")


# --- Query RAG Function with Structured Response and Caching ---
def query_insurance_rag(query_text: str, index: VectorStoreIndex, llm_model: Any) -> AnswerResponse:
    """
    Queries the RAG system using the provided index and LLM.
    Returns a structured AnswerResponse with decision, amount, and justification.
    Includes caching for faster repeated queries.
    """
    # Create cache key
    cache_key = hashlib.md5(query_text.encode()).hexdigest()
    current_time = time.time()
    
    # Check cache first
    if cache_key in response_cache:
        cached_response, timestamp = response_cache[cache_key]
        if current_time - timestamp < cache_ttl:
            print(f"--- Using Cached Response for Query: '{query_text}' ---")
            return cached_response
    
    print(f"\n--- Processing Query: '{query_text}' ---")
    query_engine = index.as_query_engine(
        llm=llm_model, 
        similarity_top_k=3,  # Reduced from 5 for speed
        response_mode="compact"  # Faster response mode
    )
    
    # Simplified prompt for faster processing
    enhanced_prompt = f"""
    Answer this insurance question: {query_text}

    Respond in JSON format:
    {{"decision": "Covered/Not Covered/Partially Covered", "amount": "amount details", "justification": "brief explanation"}}
    """
    
    response = query_engine.query(enhanced_prompt)
    print("--- Query Response Received ---")
    
    try:
        response_text = response.response.strip()
        
        # Quick JSON extraction
        if "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            response_text = response_text[start:end]
        
        response_json = json.loads(response_text)
        
        result = AnswerResponse(
            decision=response_json.get("decision", "Unable to determine"),
            amount=response_json.get("amount", "Amount not specified"),
            justification=response_json.get("justification", "Unable to provide clear justification")
        )
        
        # Cache the result
        response_cache[cache_key] = (result, current_time)
        return result
    
    except Exception as e:
        print(f"Error parsing response: {e}")
        fallback_result = AnswerResponse(
            decision="Unable to determine",
            amount="Amount not specified",
            justification=f"Processing error: {str(e)[:50]}..."
        )
        
        # Cache fallback too
        response_cache[cache_key] = (fallback_result, current_time)
        return fallback_result


# --- API Endpoint: POST /hackrx/run ---
@app.post("/hackrx/run", response_model=RunResponse)
async def run_submission(
        request: RunRequest,
        authorized: bool = Depends(verify_token)
):
    """
    Processes input documents (from URL) and answers a list of questions based on their content.
    Optimized for fast response times with concurrent processing.
    """
    print(f"\nAPI Request Received - Document URL: {request.documents}")
    print(f"API Request Received - Questions: {request.questions}")

    try:
        # Step 1: Fetch and process the document from the URL for this request
        current_documents = fetch_and_process_document_from_url(request.documents)

        if not current_documents:
            raise HTTPException(status_code=400, detail="Failed to load or process document from URL.")

        # Step 2: Generate a unique namespace for this document based on its URL hash
        doc_hash = hashlib.md5(request.documents.encode()).hexdigest()
        current_namespace = f"doc-{doc_hash}"

        # Step 3: Dynamically set up/connect to the Vector Store Index for THIS document's namespace
        dynamic_index = setup_vector_store_index_for_request(
            current_documents,
            collection_name="hackrx-insurance-rag",
            namespace=current_namespace
        )

        # Step 4: Process all questions concurrently for faster response
        print(f"--- Processing {len(request.questions)} questions concurrently ---")
        collected_answers = process_question_batch(request.questions, dynamic_index, gemini_llm_model)
        
        # Ensure we have the same number of answers as questions
        while len(collected_answers) < len(request.questions):
            collected_answers.append(AnswerResponse(
                decision="Error",
                amount="N/A",
                justification="Failed to process question"
            ))

        return RunResponse(answers=collected_answers)

    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        print(f"Unexpected error during API request processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# --- Main entry point to run the FastAPI server ---
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    print("Starting FastAPI server...")
    print(f"API documentation (Swagger UI) available at: http://127.0.0.1:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)