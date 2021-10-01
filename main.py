from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from classifier import classifier

load_dotenv()

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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7070, reload=True)
