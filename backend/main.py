from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predict

app = FastAPI(
    title="Investo API",
    description="Investment Portfolio Guidance — Next Day Price Prediction",
    version="2.0"
)

# CORS — allows your Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the prediction router
app.include_router(predict.router)

@app.get("/")
def root():
    return {"status": "Investo API is running ✅"}