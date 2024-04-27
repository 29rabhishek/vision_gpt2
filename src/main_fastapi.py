from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import cv2
import numpy as np

app = FastAPI()

# Example function for model inference (replace with your actual function)
def get_caption(img):
    # Placeholder code for model inference
    return f"Caption for {img}"

@app.post("/imagespeak/predict")
async def predict_image(ktp_image: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await ktp_image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Assess image quality using OpenCV (cv2)
        gray_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        norm_q_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()  # Example quality assessment

        print(norm_q_score)

        # Save the image if necessary (for debugging or other reasons)
        # path = Path(os.getcwd())/"test.png"
        # pil_image.save(path)

        if norm_q_score < 0.6:
            caption = "Image is too blurry"
        else:
            caption = get_caption(pil_image)

        result = {"Result": {"results": caption}}
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
