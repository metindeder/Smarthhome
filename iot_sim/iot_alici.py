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
        # İlgili kanallara abone ol
        await client.subscribe("ev/salon/komutlar")
        await client.subscribe("ev/komutlar/ai")
        print("IoT Cihazı: Dinleniyor (ev/salon/komutlar + ev/komutlar/ai)...")

        # Gelen mesajları dinle
        async for mesaj in client.messages:
            # Binary veriyi MessagePack ile çöz
            cozulmus_veri = msgpack.unpackb(mesaj.payload)
            
            topic = str(mesaj.topic)
            
            if topic == "ev/komutlar/ai":
                # Yeni AI format: {cihaz, aksiyon, deger}
                cihaz = cozulmus_veri.get('cihaz', '?')
                aksiyon = cozulmus_veri.get('aksiyon', '?')
                deger = cozulmus_veri.get('deger', None)
                deger_str = f" → {deger}" if deger is not None else ""
                print(f"AI KOMUT: {cihaz} → {aksiyon}{deger_str}")
            else:
                # Eski format: {cihaz, durum, parlaklik}
                print(f"EYLEM GERÇEKLEŞTİ: {cozulmus_veri['cihaz']} durumu {cozulmus_veri['durum']} yapıldı. (Parlaklık: {cozulmus_veri['parlaklik']})")

if __name__ == "__main__":
    asyncio.run(main())