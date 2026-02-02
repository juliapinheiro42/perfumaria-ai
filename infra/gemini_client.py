import os
from groq import Groq
from dotenv import load_dotenv

class GeminiClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            print("\n AVISO: 'GROQ_API_KEY' não encontrada no .env")
            self.client = None
            return

        try:
            self.client = Groq(api_key=self.api_key)
            self.model = "llama-3.3-70b-versatile" 
            print(" Cliente Groq inicializado (Llama 3.3 70B)")
        except Exception as e:
            print(f"Erro ao iniciar Groq: {e}")
            self.client = None

    def generate(self, prompt: str) -> str:
        if not self.client:
            return '{"error": "Cliente Groq não configurado"}'

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that outputs only JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.5,
                response_format={"type": "json_object"}, 
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            print(f"\n❌ Erro na Groq API: {e}")
            return f'{{"error": "{str(e)}"}}'