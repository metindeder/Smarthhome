import msgpack

class MQTTServisi:
    def __init__(self):
        # Başlangıçta boş. FastAPI ayağa kalkarken bunu dolduracak.
        self.client = None

    async def komut_gonder(self, topic: str, veri: dict):
        if not self.client:
            print("⚠️ Hata: MQTT bağlantısı aktif değil!")
            return

        # Veriyi MsgPack ile ultra hızlı binary formata sıkıştır
        binary_veri = msgpack.packb(veri)
        
        # QoS 0 ile asenkron olarak NanoMQ'ya fırlat
        await self.client.publish(topic, payload=binary_veri, qos=0)

mqtt_client = MQTTServisi()