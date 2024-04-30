from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
from pathlib import Path
from infer import create_vision_gpt_model
from infer_vision_gpt2 import create_vision_gpt_hf
from utils import assess_image_quality
import numpy as np
import cv2
app = FastAPI()

# Define a global caption generator instance
class CaptionGenerator:
    def __init__(self, model_name, model_path=None):
        if model_name == "VisionGPT":
            if model_path is not None:
                self.model = create_vision_gpt_model(model_path)
            else:
                raise {"model path is None"}
        if model_name == "VisionGPT_hf":
            self.model = create_vision_gpt_hf()

    def get_caption(self, img):
        return self.model.generate_caption(img)
    


# Initialize the caption generator
MODEL_NAME = "VisionGPT_hf"
# MODEL_PATH = "/media/arnav/MEDIA/Abhishek/vision_gpt2/captioner/captioner_av_10s.pt"
caption_genrator = CaptionGenerator(model_name=MODEL_NAME)


@app.get("/")
def read_root():
    return {"result": "root"}


# Endpoint to receive an image file and perform inference

@app.post("/imagespeak/predict")
async def predict_image(ktp_image: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await ktp_image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # image quality assessment
        gray_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        norm_q_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()  # Example quality assessment
        # Save the image if necessary (for debugging or other reasons)
        # path = Path(os.getcwd())/"test.png"
        # pil_image.save(path)

        if norm_q_score < 0.6:
            caption = "Image is too blurry"
        else:
            caption = caption_genrator.get_caption(pil_image)[0]



        #test
        result = {
            "Result": {
                "results": caption
            }
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


