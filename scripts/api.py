import os
from fastapi import FastAPI, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import sys
import io
import base64

dirname = os.path.abspath(os.path.join(__file__, "..", "..", "lib"))
sys.path.append(dirname)

app = FastAPI()

def pil_image_to_base64_url(image: Image.Image, format: str = "PNG") -> str:
    # Save image to an in-memory buffer
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    # Convert the image to a base64 string
    image_bytes = buffer.getvalue()
    base64_str = base64.b64encode(image_bytes).decode("utf-8")

    # Create a data URL for the image
    return f"data:image/{format.lower()};base64,{base64_str}"

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
if not os.path.exists(model_folder_path):
    # Create the folder (and any necessary intermediate directories)
    os.makedirs(model_folder_path)

@app.get("/models")
async def get_models():
    import torch
    models = [f for f in os.listdir(model_folder_path) if os.path.isfile(os.path.join(model_folder_path, f))]
    return {"models": models, "cuda": torch.cuda.is_available()}

loaded_model = None
model = None

@app.post("/process")
async def process_images(content_image: UploadFile, model_name: str = Form()):
    global loaded_model
    global model
    content_image_raw = await content_image.read()

    from utils import preprocess, deprocess, resize_image_with_max_resolution
    from DeepStyleX import DeepStyleX
    import torch

    # Open the image using PIL
    image = Image.open(io.BytesIO(content_image_raw)).convert('RGB')

    if not torch.cuda.is_available():
        image = resize_image_with_max_resolution(image, max_resolution=200000)

        
    input_image = preprocess(image, resize=False).unsqueeze(0)
    

    try:
        if loaded_model!=model_name:
            loaded_model=model_name
            model, _ = DeepStyleX.load(os.path.join(model_folder_path, model_name))
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not found",
        )
    output = model(input_image)
    output_image = deprocess(output)

    # Convert PIL image to a base64 URL
    image_url = pil_image_to_base64_url(output_image)

    # Return the styled image
    return JSONResponse(content={"styled_image_url": image_url})  # Replace with actual image

