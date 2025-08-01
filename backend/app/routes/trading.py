# backend/app/routes/trading.py
from fastapi import APIRouter, Query
from app.services.market_data import get_price_history, get_prediction

router = APIRouter()

@router.get("/price-history")
async def price_history(
    asset: str = Query(...),
    days: int = Query(1),
    interval: str = Query("hourly")
):
    return await get_price_history(asset, days=days, interval=interval)

@router.get("/prediction")
async def prediction(asset: str = Query(...)):
    return await get_prediction(asset)
