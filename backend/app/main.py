# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import trading
from app.services.market_data import load_price_cache

load_price_cache()


app = FastAPI(title="AI Trading API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trading.router, prefix="/api/trade")

@app.get("/")
def root():
    return {"message": "AI Trading Backend"}
