from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.core.mqtt import mqtt_client

router = APIRouter()

class KomutIstegi(BaseModel):
    cihaz: str = Field(..., description="Kontrol edilecek cihazın adı", example="salon_isik")
    durum: int = Field(..., description="1: Açık, 0: Kapalı", example=1)
    parlaklik: int = Field(100, description="0-100 arası parlaklık değeri")

@router.post("/kontrol")
async def cihaza_komut_gonder(istek: KomutIstegi):
    veri = istek.model_dump()
    
    # DİKKAT: await geri geldi!
    await mqtt_client.komut_gonder("ev/salon/komutlar", veri)
    
    return {
        "durum": "basarili",
        "mesaj": "Komut NanoMQ'ya asenkron iletildi",
        "iletilen_veri": veri
    }