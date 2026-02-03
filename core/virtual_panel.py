import json


class VirtualPanel:
    def __init__(self, gemini_client):
        self.client = gemini_client

    def simulate_perception(self, formula_data):
        prompt = f"""
        Atue como um painel sensorial de 50 especialistas. 
        Analise a seguinte composição de perfume: {formula_data}
        
        Com base em associações culturais e aromacologia, preveja a percepção:
        Responda APENAS em JSON:
        {{
            "vibe_principal": "string",
            "score_agradabilidade": 0-10,
            "percepcao_publico": "ex: 85% associam a energia"
        }}
        """
        response = self.client.generate(prompt)
        return json.loads(response)