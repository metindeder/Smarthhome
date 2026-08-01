import sys
import asyncio
import aiomqtt
import msgpack
import time

# WINDOWS İÇİN KRİTİK ASENKRON YAMASI
if sys.platform.lower() == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():
    # Modelden çıkan karar (Normalde bu bir JSON metni olurdu)
    ai_karari = {
        "cihaz": "salon_isik",
        "durum": 1,         # 1: Açık, 0: Kapalı
        "parlaklik": 85
    }

    # Veriyi MessagePack ile Binary'ye çevir (Çok Hızlı)
    paket = msgpack.packb(ai_karari)

    # NanoMQ Broker'a bağlan (localhost)
    async with aiomqtt.Client("127.0.0.1", port=1883) as client:
        print(f"Merkez: Komut gönderiliyor... Boyut: {len(paket)} bytes")
        # QoS 0 (Hız için) ile yayınla
        await client.publish("ev/salon/komutlar", payload=paket, qos=0)
        print("Merkez: Komut NanoMQ'ya basıldı.")

if __name__ == "__main__":
    asyncio.run(main())