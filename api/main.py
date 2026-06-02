from fastapi import FastAPI, UploadFile, File, HTTPException, status
import time

app = FastAPI(
    title="Distributed CV Engine API",
    description="Asynchronous ingestion layer for computer vision and 3D geometric processing pipelines.",
    version="1.0.0"
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Verifies the operational status of the API instance.
    """
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.post("/api/v1/ingest/image", status_code=status.HTTP_202_ACCEPTED)
async def ingest_image(file: UploadFile = File(...)):
    """
    Receives incoming image file streams asynchronously.
    Validates file extension types before processing.
    """
    allowed_extensions = ["jpg", "jpeg", "png"]
    file_extension = file.filename.split(".")[-1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed formats: {allowed_extensions}"
        )
    
    # Read the file payload asynchronously without blocking the event loop
    file_bytes = await file.read()
    payload_size_bytes = len(file_bytes)
    
    # Log information regarding receipt of the payload
    print(f"Received file: {file.filename} | Size: {payload_size_bytes} bytes")
    
    return {
        "filename": file.filename,
        "size_bytes": payload_size_bytes,
        "status": "queued_for_processing"
    }