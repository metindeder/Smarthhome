"""
iot_sim/sanal_cihazlar.py — Akıllı Ev IoT Simülatörü
MQTT broker'ı dinler ve gelen komutlara göre sanal cihazların durumunu ekrana basar.
"""

import json
import paho.mqtt.client as mqtt

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883

def on_connect(client, userdata, flags, rc):
    print(f"Sanal Akilli Ev Sistemi Broker'a Baglandi! (Durum Kodu: {rc})")
    # MQTT'de '+' işareti wildcard (joker) anlamına gelir. 
    # ev/+/komut diyerek ev/TV/komut, ev/klima/komut gibi tüm cihazları tek seferde dinliyoruz.
    client.subscribe("ev/+/komut")
    print("Tüm cihazlar dinlemede (Topic: ev/+/komut)...\n")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        # --- ŞU SATIRI EKLE ---
        print(f"DEBUG: Gelen cihaz: '{payload.get('cihaz')}', Gelen aksiyon: '{payload.get('aksiyon')}'")
        # ----------------------
        cihaz = payload.get("cihaz", "Bilinmeyen Cihaz")
        # Gelen byte verisini JSON'a çeviriyoruz
        payload = json.loads(msg.payload.decode('utf-8'))
        cihaz = payload.get("cihaz", "Bilinmeyen Cihaz")
        aksiyon = payload.get("aksiyon", "Bilinmeyen Aksiyon")
        deger = payload.get("deger")

        print("="*50)
        print(f"📡 [SİNYAL ALINDI] Topic: {msg.topic}")
        print(f"🔌 Cihaz   : {cihaz.upper()}")
        print(f"🛠️ Aksiyon : {aksiyon.upper()}")
        if deger is not None:
            print(f"🌡️ Değer   : {deger}")
        
        # Cihazın fiziksel tepkisini simüle et
        if aksiyon == "ac":
            print(f"✅ DURUM   : {cihaz.upper()} başarıyla AÇILDI.")
        elif aksiyon == "kapat":
            print(f"❌ DURUM   : {cihaz.upper()} başarıyla KAPATILDI.")
        elif aksiyon == "ayarla":
            print(f"⚙️ DURUM   : {cihaz.upper()} AYARLANDI. (Değer: {deger})")
        elif aksiyon == "azalt":
            print(f"📉 DURUM   : {cihaz.upper()} seviyesi AZALTILDI.")
        elif aksiyon == "durdur":
            print(f"⏸️ DURUM   : {cihaz.upper()} DURAKLATILDI.")
        elif aksiyon == "baslat":
            print(f"▶️ DURUM   : {cihaz.upper()} ÇALIŞTIRILDI.")

        # Teyit Mesajı Gönderimi
        try:
            durum_topic = f"ev/{cihaz.replace(' ', '_')}/durum"
            durum_payload = json.dumps({"durum": "basarili", "cihaz": cihaz, "aksiyon": aksiyon}, ensure_ascii=False)
            client.publish(durum_topic, durum_payload)
            print(f"📤 [TEYİT GÖNDERİLDİ] -> Topic: {durum_topic}")
        except Exception as pub_err:
            print(f"⚠️ [TEYİT HATASI]: {pub_err}")


        print("="*50 + "\n")

    except json.JSONDecodeError:
        print(f"⚠️ [HATA] Gelen veri JSON formatında değil: {msg.payload}")
    except Exception as e:
        print(f"⚠️ [HATA] Mesaj işlenirken bir sorun oluştu: {e}")

if __name__ == "__main__":
    client = mqtt.Client(client_id="iot_simulator")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever() # Sürekli dinlemede kal
    except Exception as e:
        print(f"Broker'a bağlanılamadı. Mosquitto açık mı? Detay: {e}")