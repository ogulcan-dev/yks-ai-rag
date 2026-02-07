# YKS AI RAG

YKS (Yükseköğretim Kurumları Sınavı) öğrencileri için geliştirilmiş, RAG (Retrieval Augmented Generation) tabanlı soru çözüm ve konu anlatım asistanı.

## Özellikler

- **Yerel Doküman İşleme**: PDF ve TXT formatındaki ders notlarını ve kitapları işler.
- **Vektör Arama**: FAISS kullanarak hızlı ve alakalı içerik, formül ve örnek soru bulur.
- **Akıllı Çözüm**: Google Gemini API (Gemini 1.5) kullanarak adım adım, anlaşılır matematiksel çözümler üretir.
- **Hızlı ve Hafif**: GPU gerektirmez, CPU üzerinde çalışabilir.

## Kurulum

1. **Gereksinimleri Yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

2. **Çevresel Değişkenleri Ayarlayın**
   `.env.example` dosyasını `.env` olarak kopyalayın ve Gemini API anahtarınızı ekleyin.
   ```bash
   copy .env.example .env
   ```
   `.env` dosyasını açıp `GEMINI_API_KEY` değerini girin.
   
   Ayrıca `HF_TOKEN` (Hugging Face Token) eklemeniz önerilir (https://huggingface.co/settings/tokens):
   ```
   HF_TOKEN=hf_...
   ```

## Kullanım

### 1. Doküman Ekleme ve İndeksleme
`documents/` klasörüne PDF veya TXT dosyalarınızı atın. Örnek olarak `konu_anlatimi_ornek.txt` dosyası eklenmiştir.

İndeksleme işlemini başlatmak için:
```bash
python -m ingest.ingest_documents
```
Bu işlem dokümanları parçalara böler (chunking), embedding'lerini çıkarır ve `index/` klasörüne kaydeder.

### 2. Backend'i Başlatma
API sunucusunu başlatın:
```bash
uvicorn app.main:app --reload
```
Sunucu `http://localhost:8000` adresinde çalışacaktır.

### 3. Soru Sorma
API çalışırken bir POST isteği göndererek soru sorabilirsiniz.

**Örnek İstek (Curl):**
```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"x^2 + 5x + 6 = 0 kökleri nedir?\"}"
```

**Örnek İstek (Python):**
```python
import requests

url = "http://localhost:8000/ask"
payload = {"question": "x^2 + 5x + 6 = 0 kökleri nedir?"}
response = requests.post(url, json=payload)
print(response.json())
```

## Proje Yapısı
- `app/`: Ana uygulama kodu (API, Core, Utils)
- `ingest/`: Doküman işleme script'leri
- `documents/`: Kaynak dokümanlar
- `index/`: Vektör veritabanı dosyaları
