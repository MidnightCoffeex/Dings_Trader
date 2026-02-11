import sys
import os
import asyncio
from fastapi import UploadFile
import io

# Add ml to path
sys.path.append(os.path.join(os.getcwd(), "ml"))

from api import upload_model_package
from db import init_db, list_model_packages

async def smoke_test_upload():
    print("Initializing DB...")
    init_db()
    
    # Create dummy files
    forecast_content = b"dummy forecast content"
    ppo_content = b"dummy ppo content"
    
    forecast_file = UploadFile(
        filename="test.pt",
        file=io.BytesIO(forecast_content)
    )
    ppo_file = UploadFile(
        filename="test.zip",
        file=io.BytesIO(ppo_content)
    )
    
    print("Testing upload_model_package function...")
    try:
        result = await upload_model_package(
            name="Smoke Test Model",
            forecast_model=forecast_file,
            ppo_model=ppo_file
        )
        print("Upload Result:", result)
        
        # Verify in DB
        packages = list_model_packages()
        found = any(p["id"] == result["id"] for p in packages)
        print(f"Verified in DB: {found}")
        
        if found:
            print("PASS: Upload and DB registry working.")
        else:
            print("FAIL: Model not found in DB after upload.")
            
    except Exception as e:
        print(f"FAIL: Exception during upload test: {e}")

if __name__ == "__main__":
    asyncio.run(smoke_test_upload())
