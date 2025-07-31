import subprocess
import sys
import os

# Install requirements if needed
try:
    import fastapi
    import uvicorn
    import google.generativeai
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Import the minimal API
from api_minimal import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Starting HackRx Insurance RAG API on port {port}")
    print(f"📖 API Documentation: http://localhost:{port}/docs")
    print(f"🔗 API Endpoint: http://localhost:{port}/hackrx/run")
    uvicorn.run(app, host="0.0.0.0", port=port)
