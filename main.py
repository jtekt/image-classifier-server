from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from classifier import Classifier
import zipfile
import io

classifier = Classifier()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
    "application_name": "image classifier server",
    "author": "Maxime MOREILLON",
    "version": "0.2.3",
    "model_loaded": classifier.model_loaded,
    'model_info': {**classifier.model_info}
    }

@app.post("/predict")
async def predict(image: bytes = File()):
    result = await classifier.predict(image)
    return result

@app.post("/model")
async def upload_model(model: bytes = File()):
    fileBuffer = io.BytesIO(model)
    with zipfile.ZipFile(fileBuffer) as zip_ref:
        zip_ref.extractall('./model')
        classifier.load_model()
        return {"file_size": len(model)}