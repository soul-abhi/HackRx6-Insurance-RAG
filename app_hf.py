import subprocess
import sys
import os

# Install requirements
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_fast.txt"])
except:
    print("Installing basic requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "google-generativeai", "requests", "PyMuPDF"])

# Import the fast API
from api_fast import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
