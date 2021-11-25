from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from classifier import Classifier

classifier = Classifier()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
    "application_name": "image classifier (Fast API)",
    "author": "Maxime MOREILLON",
    }


@app.post("/predict")
async def predict(image: UploadFile = File (...)):
    result = await classifier.predict(image)
    return result
