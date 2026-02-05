# Perfumaria AI

Bem-vindo ao **Perfumaria AI**, um sistema avançado de Inteligência Artificial projetado para simular a descoberta e otimização de fórmulas de perfumes. Este projeto combina o poder das **Redes Neurais Profundas (Deep Learning)**, **Otimização Bayesiana** e **Modelos de Linguagem (LLMs)** para criar, avaliar e refinar blends moleculares complexos.

O objetivo é acelerar o processo de P&D (Pesquisa e Desenvolvimento) na perfumaria, identificando combinações promissoras de moléculas que atendam a critérios de estabilidade, custo e perfil olfativo.

---

## Arquitetura e Funcionamento

O sistema opera em um ciclo contínuo de aprendizado e descoberta, dividido em quatro pilares principais:

### 1. Geração de Dados e Dataset
- O sistema inicia criando um dataset sintético (Warmup) baseado em propriedades químicas reais (volatilidade, peso molecular, polaridade) a partir de um banco de insumos (`insumos.csv`).
- As moléculas são codificadas em vetores numéricos (`core/encoder.py`) para serem processadas pelos modelos.

### 2. Modelagem Preditiva (`core/model.py`)
- Uma **Rede Neural (PyTorch)** é treinada para prever o "sucesso" de uma fórmula.
- A arquitetura é um Multi-Layer Perceptron (MLP) que recebe as características químicas e estima uma pontuação de fitness.
- *Nota:* Comecei usando implementações manuais em numpy, mas assim que decidi aumentar o projeto vi que a IA estava ficando confiante nas previsões então criei um dataset novo e usei pytorch no modelo para uma previsão mais eficiente.

### 3. Estratégia Cognitiva (`core/strategy.py`)
- Um **Agente de Estratégia** utiliza LLMs (via `infra/llm_client.py`) para analisar o histórico de descobertas.
- O LLM (Llama 3.3 70B via Groq) atua como um "Perfume Master", sugerindo ajustes nos parâmetros de busca (ex: "aumentar a volatilidade média" ou "reduzir o custo") com base nos resultados anteriores.

### 4. Motor de Descoberta e Otimização (`core/discovery.py` & `core/surrogate.py`)
- O motor de descoberta orquestra o ciclo de inovação.
- Utiliza um **Modelo Substituto Bayesiano (Gaussian Process)** para selecionar os candidatos mais promissores.
- Aplica a técnica de **Expected Improvement (EI)** para balancear a exploração de novas áreas químicas (incerteza alta) com a explotação de áreas conhecidas (sucesso alto).

---

##  Instalação e Configuração

### Pré-requisitos
- Python 3.10 ou superior
- `pip` (gerenciador de pacotes)
- Chave de API da Groq (necessária para o agente de estratégia)

### Passo a Passo

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/jaif/perfumaria-ai.git
    cd perfumaria-ai
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuração de Variáveis de Ambiente:**
    Crie um arquivo `.env` na raiz do projeto e adicione sua chave de API da Groq:
    ```env
    GROQ_API_KEY=sua-chave-aqui
    ```
    Obtenha sua chave em: https://console.groq.com/

---

## Como Usar

Para iniciar a simulação de descoberta de perfumes, primeiro execute o turning, após isso execute o arquivo principal:

```bash
python -m experiments.tuning
streamlit run main.py
```

### O que acontece durante a execução?
1.  **Warmup**: Se não houver modelo salvo, 200 amostras sintéticas são geradas e o modelo é treinado inicialmente.
2.  **Ciclo de Descoberta**: O sistema executa 30 rodadas de otimização.
    - A cada rodada, o agente (LLM) analisa os dados e propõe estratégias.
    - O motor de busca gera novos candidatos.
    - O modelo avalia e salva as melhores descobertas.
3.  **Resultados**: As descobertas são salvas em `results/discoveries.json` e exibidas no console.

---

## 📂 Estrutura do Projeto

```
/
├── core/                   # Núcleo da lógica do sistema
│   ├── chemistry.py        # Simulação de propriedades químicas
│   ├── discovery.py        # Motor de otimização e loop principal
│   ├── encoder.py          # Codificação de features moleculares
│   ├── market.py           # Análise de viabilidade econômica
│   ├── model.py            # Rede Neural (PyTorch)
│   ├── replay_buffer.py    # Memória de experiências
│   ├── strategy.py         # Agente de estratégia (LLM)
│   ├── surrogate.py        # Gaussian Process (Otimização Bayesiana)
│   └── trainer.py          # Utilitários de treinamento
├── experiments/            # Scripts de experimentação
├── infra/                  # Infraestrutura e integrações externas
│   └── llm_client.py       # Cliente para API de LLMs (Groq)
├── results/                # Saída dos dados gerados
├── tests/                  # Testes automatizados (estrutura prevista)
├── insumos.csv             # Base de dados de matérias-primas
├── main.py                 # Ponto de entrada da aplicação
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação
```

---



## Video funcionando 
<video controls src="Recording 2026-01-29 164248.mp4" title="Ia perfuma-ai funcionando"></video>

*Projeto desenvolvido como demonstração de IA aplicada à Química e Negócios.*
