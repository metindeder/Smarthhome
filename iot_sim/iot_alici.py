import sys
import asyncio
import aiomqtt
import msgpack

# WINDOWS İÇİN KRİTİK ASENKRON YAMASI
if sys.platform.lower() == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():

    # NanoMQ Broker'a bağlan
    async with aiomqtt.Client("127.0.0.1", port=1883) as client:
        # İlgili kanala abone ol
        await client.subscribe("ev/salon/komutlar")
        print("IoT Cihazı: Dinleniyor (ev/salon/komutlar)...")

        # Gelen mesajları dinle
        async for mesaj in client.messages:
            # Binary veriyi MessagePack ile çöz
            cozulmus_veri = msgpack.unpackb(mesaj.payload)
            
            # Eylemi gerçekleştir (Terminalde göster)
            print(f"EYLEM GERÇEKLEŞTİ: {cozulmus_veri['cihaz']} durumu {cozulmus_veri['durum']} yapıldı. (Parlaklık: {cozulmus_veri['parlaklik']})")

if __name__ == "__main__":
    asyncio.run(main())