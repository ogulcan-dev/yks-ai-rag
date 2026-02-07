import os
from google import genai
from typing import List

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_id = "gemini-2.5-flash"

    def generate_answer(self, context: str, question: str) -> str:
        """
        Generate answer using Google GenAI SDK (v1) with the specified prompt.
        """
        system_prompt = """Sen bir YKS matematik öğretmenisin.

Aşağıdaki bağlamı kullanarak soruyu çöz.

Kurallar:
- Çözümü adım adım anlat
- Gerekirse formülleri yaz
- En sonda "Cevap:" şeklinde sonucu belirt
- Türkçe cevap ver
- YKS öğrencisine uygun anlaşılır bir dil kullan
- Sadece sana verilen bağlamı kullan
- Dokümanlarda olmayan bir şey sorulursa bilmiyorum diye cevap ver
- Müfredata uygun cevaplar ver
- Örneğin dökümanlarında limitle alaklı bir tanım yok, o yüzden ona bilmiyorum diye cevap ver
- Bilmediğin bir şey sorulduğunda sona Cevap: şeklinde bir şey ekleme.
"""
        
        user_prompt = f"""{system_prompt}

Bağlam:
{context}

Soru:
{question}
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=user_prompt
            )
            return response.text
        except Exception as e:
            return f"Bir hata oluştu: {str(e)}"
