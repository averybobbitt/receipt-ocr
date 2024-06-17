from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from numpy import uint8, frombuffer

from utils import perform_ocr

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the OCR API"}


@app.post("/ocr/")
async def ocr_receipt(file: UploadFile):
    # check if the uploaded file is an image
    if file.content_type.startswith("image"):
        image_bytes = await file.read()
        img_array = frombuffer(image_bytes, uint8)
        ocr_text = perform_ocr(img_array)

        return JSONResponse(content={"result": ocr_text}, status_code=200)
    else:
        return {"error": "Uploaded file is not an image"}
