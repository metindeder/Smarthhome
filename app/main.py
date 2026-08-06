import sys
import asyncio
import aiomqtt
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Servisler ve Yönlendiriciler
from app.core.mqtt import mqtt_client
from app.api.cihaz_kontrol import router as cihaz_router

# WINDOWS ASENKRON YAMASI (Uvicorn bu satırı okuyacak ve kurallara uyacak!)
if sys.platform.lower() == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Sistem Başlatılıyor: Uvicorn & aiomqtt devrede...")
    
    # aiomqtt'nin zorunlu kıldığı güvenli ve asenkron bağlantı bloğu
    async with aiomqtt.Client("127.0.0.1", port=1883) as client:
        # Bağlanan istemciyi servise devret
        mqtt_client.client = client
        print("MQTT Bağlantısı Başarılı (Asenkron NanoMQ - 127.0.0.1:1883)")
        
        yield # FastAPI burada çalışmaya başlar
        
    print("Sistem Kapatılıyor: Tüm süreçler durduruldu.")

app = FastAPI(
    title="Edge AI Akıllı Ev Merkezi",
    lifespan=lifespan
)

app.include_router(cihaz_router, prefix="/api/v1", tags=["Cihaz Kontrolü"])

@app.get("/")
async def root():
    return {"durum": "aktif", "sunucu": "Uvicorn"}