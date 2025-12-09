from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_image
import uvicorn

app = FastAPI()

origins = ["https://covid-classifier-frontend.onrender.com"]  

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file bytes
        file_bytes = await file.read()

        # Call inference function
        result = predict_image(file_bytes)

        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
