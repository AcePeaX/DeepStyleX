import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend's URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock available models
models = ["Model A", "Model B", "Model C"]

model_folder_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'saves')

@app.get("/models")
async def get_models():
    models = [f for f in os.listdir(model_folder_path) if os.path.isfile(os.path.join(model_folder_path, f))]
    return {"models": models}

@app.post("/process")
async def process_images(content_image: UploadFile, style_image: UploadFile, model_name: str = Form()):
    # Load and process the images using the selected model
    # Replace with your processing logic
    result_image = b""  # Placeholder for processed image bytes
    return JSONResponse(content={"result_url": "data:image/png;base64,..."})  # Replace with actual image

