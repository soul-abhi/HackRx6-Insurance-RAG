# import os
#
# from typing import Any, List, Optional, Iterator, AsyncIterator
# from llama_index.core.base.llms.types import ChatResponse, LLMMetadata, CompletionResponse, CompletionResponseGen, ChatResponseGen
# from llama_index.core.llms import LLM, ChatMessage, MessageRole
#
# import google.generativeai as genai
# import os
#
#
# class CustomGeminiLLM(LLM):
#     model_name: str = "gemini-1.5-flash"
#     temperature: float = 0.7
#     max_tokens: Optional[int] = None
#     api_key: str = os.getenv("GOOGLE_API_KEY")
#
#     _client: Any = None  # Internal client instance
#
#     def __init__(self, **kwargs: Any):
#         super().__init__(**kwargs)
#         if self.api_key is None:
#             raise ValueError("GOOGLE_API_KEY must be set in your environment.")
#         genai.configure(api_key=self.api_key)
#         self._client = genai.GenerativeModel(self.model_name)
#
#     @property
#     def metadata(self) -> LLMMetadata:
#         return LLMMetadata(
#             model_name=self.model_name,
#             temperature=self.temperature,
#             max_tokens=self.max_tokens,
#         )
#
#     def _get_text_from_content(self, content: Any) -> str:
#         """Extracts text from Gemini's Content object."""
#         if content and hasattr(content, 'parts') and content.parts:
#             for part in content.parts:
#                 if hasattr(part, 'text'):
#                     return part.text
#         return ""
#
#     def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
#         """Generate a completion from a prompt."""
#         print(f"DEBUG: Before pop - kwargs in complete: {kwargs}") # <--- DEBUG PRINT
#         # --- FIX: Remove 'formatted' kwarg if present ---
#         if "formatted" in kwargs:
#             kwargs.pop("formatted")
#         print(f"DEBUG: After pop - kwargs in complete: {kwargs}")  # <--- DEBUG PRINT
#
#         response = self._client.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=self.temperature,
#                 max_output_tokens=self.max_tokens,
#             ),
#             **kwargs,
#         )
#         text = self._get_text_from_content(response.candidates[0].content)
#         return CompletionResponse(text=text, raw=response)
#
#     async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
#         return self.complete(prompt, **kwargs)
#
#     def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
#         if "formatted" in kwargs:
#             kwargs.pop("formatted") # Also filter here for consistency
#
#         response_gen = self._client.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=self.temperature,
#                 max_output_tokens=self.max_tokens,
#             ),
#             stream=True,
#             **kwargs,
#         )
#         for chunk in response_gen:
#             text = self._get_text_from_content(chunk.candidates[0].content)
#             yield CompletionResponse(text=text, raw=chunk)
#
#     async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncIterator[CompletionResponseGen]:
#         for response in self.stream_complete(prompt, **kwargs):
#             yield response
#
#     def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
#         if "formatted" in kwargs:
#             kwargs.pop("formatted") # Also filter here for consistency
#
#         gemini_messages = []
#         for message in messages:
#             role = "user" if message.role == MessageRole.USER else "model"
#             gemini_messages.append({"role": role, "parts": [{"text": message.content}]})
#
#         response = self._client.generate_content(
#             gemini_messages,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=self.temperature,
#                 max_output_tokens=self.max_tokens,
#             ),
#             **kwargs,
#         )
#
#         text = self._get_text_from_content(response.candidates[0].content)
#         return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), raw=response)
#
#     async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
#         return self.chat(messages, **kwargs)
#
#     def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponseGen:
#         if "formatted" in kwargs:
#             kwargs.pop("formatted") # Also filter here for consistency
#
#         gemini_messages = []
#         for message in messages:
#             role = "user" if message.role == MessageRole.USER else "model"
#             gemini_messages.append({"role": role, "parts": [{"text": message.content}]})
#
#         response_gen = self._client.generate_content(
#             gemini_messages,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=self.temperature,
#                 max_output_tokens=self.max_tokens,
#             ),
#             stream=True,
#             **kwargs,
#         )
#         for chunk in response_gen:
#             text = self._get_text_from_content(chunk.candidates[0].content)
#             yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), raw=chunk)
#
#     async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> AsyncIterator[ChatResponseGen]:
#         for response in self.stream_chat(messages, **kwargs):
#             yield response
#
#
#
#
# #new one
# # Python Standard Library Imports
# import os
# import json
#
# from dotenv import load_dotenv
# import fitz
#
# # LlamaIndex Core Components
# from llama_index.core import Document
# from llama_index.core import VectorStoreIndex
# from llama_index.core import Settings
# from llama_index.core.node_parser import SimpleNodeParser
#
# from llama_index.core.llms import LLM, ChatMessage, MessageRole
# from llama_index.core.base.llms.types import ChatResponse, LLMMetadata, CompletionResponse, CompletionResponseGen, ChatResponseGen
#
# # LlamaIndex Integrations
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.vector_stores.chroma import ChromaVectorStore
#
# import chromadb
#
# import google.generativeai as genai
#
# load_dotenv('config/.env')
#
#
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#
# gemini_llm_model = CustomGeminiLLM(model_name="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
#
# embed_model_name = "BAAI/bge-small-en-v1.5"
# Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
# Settings.llm = gemini_llm_model
#
# print("--- Code Chunk 1.1 Executed: All essential Libraries, Environment Variables, and LlamaIndex Global Settings Loaded ---")
#
#
#
# # 1.2 chunk
# def load_documents_from_data_folder(data_folder="data"):
#     documents = []
#
#     pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
#
#     if not pdf_files:
#         print(f"No PDF files found in '{data_folder}'. Please ensure your datasets are placed in this folder.")
#         return documents
#
#     print(f"\n--- Code Chunk 1.2 Executed: Found {len(pdf_files)} PDF files in '{data_folder}'. Starting text extraction... ---")
#
#     for pdf_file_name in pdf_files:
#         filepath = os.path.join(data_folder, pdf_file_name)
#         text = ""
#         try:
#             doc = fitz.open(filepath)
#             for page_num in range(doc.page_count):
#                 page = doc.load_page(page_num)
#                 text += page.get_text()
#             doc.close()
#
#             documents.append(Document(
#                 text=text,
#                 metadata={"file_name": pdf_file_name, "file_path": filepath}
#             ))
#             print(f" - Successfully extracted text from {pdf_file_name}")
#         except Exception as e:
#             print(f" - Error extracting text from {pdf_file_name}: {e}")
#
#     print(f"--- Code Chunk 1.2 Complete: Extracted text from {len(documents)} PDF documents ---")
#     return documents
#
# # 1.3chunk
# def setup_vector_store_index(documents, persist_dir="./chroma_db"):
#     print(f"\n--- Code Chunk 1.3 Executed: Setting up Vector Store Index (Persisting to '{persist_dir}')... ---")
#
#     db = chromadb.PersistentClient(path=persist_dir)
#
#     chroma_collection = db.get_or_create_collection("insurance_policies")
#     print("--- ChromaDB collection 'insurance_policies' created/loaded ---")
#
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#
#     # Initialize a node parser to split documents into smaller chunks
#     node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
#
#     # Parse documents into nodes
#     print("--- Parsing documents into nodes... ---")
#     nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
#
#     # NEW STEP: Explicitly generate embeddings for each node
#     # This loop ensures each node has its embedding before being added to the vector store.
#     print("--- Generating embeddings for nodes... ---")
#     if Settings.embed_model:
#         for i, node in enumerate(nodes):
#             # Get the text content of the node and generate its embedding
#             node.embedding = Settings.embed_model.get_text_embedding(node.get_content())
#             # You can add a simple progress print here if you want to see it iterate
#             # if (i + 1) % 50 == 0:
#             #     print(f"  Processed {i + 1}/{len(nodes)} embeddings.")
#     else:
#         raise ValueError("Settings.embed_model is not set. Cannot generate embeddings for nodes.")
#     print(f"--- Embeddings generated for {len(nodes)} nodes. ---")
#
#     # Now, add the nodes (which now have embeddings) to the vector store.
#     # We only add if the collection is empty to avoid duplicating data on repeated runs.
#     if chroma_collection.count() == 0:
#         print("--- Adding nodes with embeddings to ChromaDB... ---")
#         vector_store.add(nodes)
#         print(f"--- Added {chroma_collection.count()} documents (nodes) to ChromaDB. ---")
#     else:
#         print(f"--- Detected {chroma_collection.count()} existing embeddings in ChromaDB. Skipping adding nodes. ---")
#
#     # Always create the index from the vector store
#     index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
#
#     print("--- Code Chunk 1.3 Complete: Vector Store Index Created and Documents Embedded (or Loaded) ---")
#     return index
#
#
# # Code Chunk 2.1
# def query_insurance_rag(query_text, index, llm_model):
#     print(f"\n--- Code Chunk 2.1 Executed: Processing Query ---")
#
#     # Create a query engine from the index.
#     # We explicitly pass the 'llm_model' because we set up Gemini directly in Code Chunk 1.1,
#     # rather than globally via Settings.llm.
#     query_engine = index.as_query_engine(llm=llm_model)
#
#     # Query the engine with the user's question
#     response = query_engine.query(query_text)
#
#     print("--- Query Response Received ---")
#     return response.response  # Return just the text of the answer
#
#
# # UPDATED ONE
# if __name__ == "__main__":
#     print("\n--- Starting HackRx 6 RAG System: Knowledge Base Setup ---")
#
#     # Call the function to load and extract text from your PDF documents.
#     all_raw_documents = load_documents_from_data_folder(data_folder="data")
#
#     if all_raw_documents:
#         # Call the function to set up the Vector Store Index (ChromaDB).
#         insurance_index = setup_vector_store_index(all_raw_documents, persist_dir="./chroma_db")
#
#         print("\n--- Knowledge Base Setup Complete! ---")
#         print("The 'insurance_index' object is now ready for querying your policies.")
#
#         # --- NEW CODE FOR STEP 2: Querying Phase ---
#         print("\n--- Starting Querying Phase ---")
#         user_query = "What are the common exclusions in property insurance policies?"
#         print(f"User Query: \"{user_query}\"")
#
#         # Call the query function.
#         # Remember 'gemini_llm_model' was created directly in Code Chunk 1.1.
#         response_text = query_insurance_rag(user_query, insurance_index, gemini_llm_model)
#
#         print("\n--- Final Answer ---")
#         print(response_text)
#         print("\n--- Querying Phase Complete! ---")
#
#     else:
#         print(
#             "\n--- Knowledge Base Setup Failed: No documents loaded. Please check your 'data' folder and its contents. ---")
#
#     print("Process finished.")
#
#
#
#
