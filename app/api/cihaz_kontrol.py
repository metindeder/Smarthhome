from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.core.mqtt import mqtt_client

router = APIRouter()

# --- Eski format (geriye dönük uyumluluk) ---
class KomutIstegi(BaseModel):
    cihaz: str = Field(..., description="Kontrol edilecek cihazın adı", example="salon_isik")
    durum: int = Field(..., description="1: Açık, 0: Kapalı", example=1)
    parlaklik: int = Field(100, description="0-100 arası parlaklık değeri")

@router.post("/kontrol")
async def cihaza_komut_gonder(istek: KomutIstegi):
    veri = istek.model_dump()
    
    await mqtt_client.komut_gonder("ev/salon/komutlar", veri)
    
    return {
        "durum": "basarili",
        "mesaj": "Komut NanoMQ'ya asenkron iletildi",
        "iletilen_veri": veri
    }

# --- Yeni AI format (asistan_beyin çıktısıyla uyumlu) ---
class AIKomutIstegi(BaseModel):
    cihaz: str = Field(..., description="Cihaz adı (TV, lamba, klima, muzik...)", example="TV")
    aksiyon: str = Field(..., description="ac, kapat, artir, azalt, ayarla, durdur, baslat", example="ac")
    deger: Optional[int] = Field(None, description="Sayısal değer (opsiyonel)", example=22)

@router.post("/ai-komut")
async def ai_komut_gonder(istek: AIKomutIstegi):
    """AI asistan pipeline'ından gelen komutları MQTT'ye iletir."""
    veri = istek.model_dump()
    
    await mqtt_client.komut_gonder("ev/komutlar/ai", veri)
    
    return {
        "durum": "basarili",
        "mesaj": "AI komutu NanoMQ'ya iletildi",
        "iletilen_veri": veri
    }