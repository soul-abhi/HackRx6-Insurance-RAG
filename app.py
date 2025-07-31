import subprocess
import sys
import os
from pathlib import Path

# Install requirements
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Import your FastAPI app
from api_server import app

# For Hugging Face Spaces
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
