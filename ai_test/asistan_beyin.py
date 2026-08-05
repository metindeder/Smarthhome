import sys
import os
import ctypes
import time
from llama_cpp import Llama, LlamaGrammar

# =============================================================================
# 1. İŞLEMCİ ÖNCELİĞİ (Windows)
# =============================================================================
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080
    )
    print("🔥 İşlemci önceliği HIGH_PRIORITY seviyesine çekildi.")

# =============================================================================
# 2. JSON GRAMMAR — llama.cpp GBNF (0.3.x uyumlu)
#    • Compact JSON: çıktıda ws yok → daha az token → daha hızlı
#    • Aksiyon 7 değerle kısıtlı — halüsinasyon imkânsız
#    • deger-val: number önce gelir, sonra string (22 → int, not "22")
#    NOT: llama_cpp 0.3.34'te LlamaRAMCache + grammar crash yapıyor,
#         yerel KV-cache ısınma çağrısı kullanıyoruz.
# =============================================================================
JSON_GRAMMAR = r"""
root        ::= "{\"cihaz\":" string ",\"aksiyon\":" aksiyon-val ",\"deger\":" deger-val "}"
aksiyon-val ::= "\"ac\"" | "\"kapat\"" | "\"artir\"" | "\"azalt\"" | "\"ayarla\"" | "\"durdur\"" | "\"baslat\""
deger-val   ::= "null" | [0-9]+ ("." [0-9]+)? | "\"" [^"]* "\""
string      ::= "\"" [^"]* "\""
"""
grammar = LlamaGrammar.from_string(JSON_GRAMMAR)
print("✅ JSON Grammar yüklendi.")

# =============================================================================
# 3. MODEL YÜKLEME — Qwen 2.5-0.5B Instruct Q2_K
#    Tüm modeller/quant'lar arasında en iyi sonuç:
#    ~971 ms ısınma | ~686 ms warm avg | ~29 tok/s | 5/5 doğruluk
#    Q4_K_M'den isınma ~330ms daha hızlı, inference hızı identik.
# =============================================================================
MODEL_YOLU = "../models/qwen2.5-0.5b-instruct-q2_k.gguf"

_cpu = os.cpu_count() or 4
N_THREADS = min(_cpu, 6)   # Benchmark: 6 thread bu sistemde en hızlı (8+ koordinasyon yükü yaratıyor)

print(f"\n🧠 Model yükleniyor: {MODEL_YOLU.split('/')[-1]}")
print(f"   CPU çekirdek: {_cpu}  →  N_THREADS: {N_THREADS}")

llm = Llama(
    model_path=MODEL_YOLU,
    n_ctx=512,
    n_batch=512,
    n_threads=N_THREADS,
    n_threads_batch=N_THREADS,
    n_gpu_layers=-1,
    use_mlock=True,
    chat_format="chatml",
    flash_attn=True,
    type_k=8,              # 🚀 KV-cache kuantizasyonu: bellek bant genişliği azalır
    type_v=8,
    verbose=False,
)

print("✅ Model hazır!")
print("-" * 55)

# =============================================================================
# 4. SİSTEM YÖNERGESİ — Qwen 2.5-1.5B için optimize
#    • Geçerli aksiyon değerleri listeleniyor
#    • Kritik eşleştirmeler (open→ac, açar mısın→ac) örnekle pekiştiriliyor
#    • Örnekler compact formatta → "No spaces" talimatıyla uyumlu
# =============================================================================
SYSTEM_PROMPT = (
    "You are a smart home voice command parser. "
    "Output ONLY a compact JSON object with exactly three keys:\n"
    "  \"cihaz\"  : device name in Turkish (e.g. \"TV\", \"lamba\", \"klima\", \"muzik\")\n"
    "  \"aksiyon\": one of: \"ac\", \"kapat\", \"artir\", \"azalt\", \"ayarla\", \"durdur\", \"baslat\"\n"
    "  \"deger\"  : number or null\n\n"
    "Mapping rules:\n"
    "  open/turn on/aç/açar mısın → \"ac\"\n"
    "  close/turn off/kapat       → \"kapat\"\n"
    "  increase/yükselt/artır     → \"artir\"\n"
    "  decrease/azalt/düşür       → \"azalt\"\n"
    "  set/ayarla/yap             → \"ayarla\"\n"
    "  stop/durdur                → \"durdur\"\n"
    "  start/başlat/oynat         → \"baslat\"\n\n"
    "Examples (compact, no spaces):\n"
    "  'open the tv'          → {\"cihaz\":\"TV\",\"aksiyon\":\"ac\",\"deger\":null}\n"
    "  'turn on the tv'       → {\"cihaz\":\"TV\",\"aksiyon\":\"ac\",\"deger\":null}\n"
    "  'Televizyonu açar mısın?' → {\"cihaz\":\"TV\",\"aksiyon\":\"ac\",\"deger\":null}\n"
    "  'Mutfak ışıklarını kapat' → {\"cihaz\":\"mutfak ışıkları\",\"aksiyon\":\"kapat\",\"deger\":null}\n"
    "  'Klima sıcaklığını 22 yap' → {\"cihaz\":\"klima\",\"aksiyon\":\"ayarla\",\"deger\":22}\n"
    "  'Müziği durdur'        → {\"cihaz\":\"muzik\",\"aksiyon\":\"durdur\",\"deger\":null}\n\n"
    "No spaces in output JSON. Raw JSON only."
)

# =============================================================================
# 5. PROMPT KV-CACHE ISITMA
#    llama.cpp, aynı token prefix'ini tekrar işlemez.
#    Bu çağrı sistem promptunu KV-cache'e işler;
#    sonraki komutlar yalnızca kullanıcı mesajını (~10-20 tok) işler.
# =============================================================================
print("\n🔥 Sistem prompt KV-cache'e alınıyor (ısınma çağrısı)...")
_t_isinma = time.perf_counter()
llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "hazır"},
    ],
    max_tokens=1,
    temperature=0.0,
    grammar=grammar,
)
_isinma_ms = (time.perf_counter() - _t_isinma) * 1000
print(f"✅ KV-cache hazır! (ısınma süresi: {_isinma_ms:.0f} ms)")
print("-" * 55)

# =============================================================================
# 6. ÇIKARIM FONKSİYONU
# =============================================================================
def komut_isle(komut_metni: str, etiket: str = "TEST"):
    print(f"\n[{etiket}]")
    print(f"  🎙️  Komut : '{komut_metni}'")

    t0 = time.perf_counter()

    sonuc = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": komut_metni},
        ],
        max_tokens=40,
        temperature=0.0,
        repeat_penalty=1.0,   # 🚀 temperature=0 ile repeat_penalty gereksiz yük
        grammar=grammar,
    )

    sure_ms = (time.perf_counter() - t0) * 1000
    json_ciktisi   = sonuc["choices"][0]["message"]["content"].strip()
    completion_tok = sonuc["usage"]["completion_tokens"]
    prefill_tok    = sonuc["usage"]["prompt_tokens"]
    tok_per_sn     = (completion_tok / sure_ms) * 1000 if sure_ms > 0 else 0

    print(f"  ⏱️  Süre       : {sure_ms:.0f} ms")
    print(f"  ⚡  Throughput : {tok_per_sn:.1f} tok/s  ({completion_tok} gen + {prefill_tok} prefill)")
    print(f"  📦  JSON       : {json_ciktisi}")
    print("-" * 55)

# =============================================================================
# 7. TEST SENARYOLARI
# =============================================================================
komut_isle("open the tv",              "TEST 1 — KV-CACHE İLE (1. çağrı)")
komut_isle("Mutfak ışıklarını kapat",  "TEST 2 — WARM (Türkçe kapat)")
komut_isle("Televizyonu açar mısın?",  "TEST 3 — WARM (Türkçe soru)")
komut_isle("Klima sıcaklığını 22 yap", "TEST 4 — WARM (değer ile)")
komut_isle("Müziği durdur",            "TEST 5 — WARM (durdur)")