# Perfumaria AI 🧪✨

**Sistema de Inteligência Artificial para Design Molecular e Otimização de Fragrâncias**

O **Perfumaria AI** é uma plataforma "End-to-End" que simula o papel de um Perfumista Master e de um Painel de Avaliação Sensorial. O sistema combina **Algoritmos Genéticos**, **Redes Neurais em Grafos (GNNs)**, **Otimização Bayesiana** e **LLMs** para descobrir, avaliar e refinar fórmulas de perfumes, equilibrando criatividade olfativa, viabilidade comercial e sustentabilidade. A plataforma conta com 200 insumos de origem Amazônica, 100% sustentável e renovável.

---

## 🚀 Funcionalidades Principais

### 🧠 Inteligência Híbrida

- **Geração Evolutiva:** Utiliza Algoritmos Genéticos para evoluir fórmulas através de cruzamento (crossover) e mutação, imitando a seleção natural de acordes bem-sucedidos (`core/evolution.py`).
- **Graph Neural Networks (GNN):** Uma rede neural baseada em PyTorch Geometric (`core/model.py`) que analisa a estrutura molecular (grafos) para prever a performance olfativa.
- **Estratégia via LLM:** Um agente cognitivo (Llama 3 via Groq) analisa o histórico de descobertas e sugere estratégias de alto nível (ex: "Aumentar volatilidade no topo") (`core/strategy.py`).
- **Otimização Bayesiana:** Utiliza Processos Gaussianos (`core/surrogate.py`) para guiar a exploração do espaço químico de forma eficiente.

### ⚗️ Simulação Físico-Química (`core/chemistry.py`)

- **Curva de Evaporação 4D:** Simula a volatilidade dos ingredientes ao longo do tempo (0h a 10h), calculando a evolução da fragrância.
- **Cálculo de Projeção (Sillage):** Estima a potência de difusão baseada em Pressão de Vapor e OAV (Odor Activity Value).
- **Neuro-Impacto:** Mapeia a fórmula em coordenadas de Valence/Arousal (Modelo de Russell) para prever efeitos emocionais (ex: Relaxamento, Energia).

### 🌍 Sustentabilidade e Compliance (`core/compliance.py`)

- **Verificação IFRA:** Checagem automática de limites de segurança regulatória.
- **Eco-Score:** Cálculo dinâmico de pegada de carbono, biodegradabilidade e renovabilidade.
- **Reformulação Verde:** Algoritmo capaz de substituir ingredientes sintéticos ou poluentes por alternativas "bio" sem alterar o perfil olfativo (`reformulate_green`).

### 💼 Inteligência de Mercado (`core/market.py`)

- **Fit Regional:** Avalia a adequação da fórmula para mercados específicos (Ásia, LatAm, Oriente Médio, EUA) baseando-se em preferências culturais.
- **Precificação Dinâmica:** Estimativa de custo fabril (Juice Cost), margem bruta e sugestão de tier de mercado (Mass, Prestige, Luxury).

---

## 🛠️ Stack Tecnológico

- **Linguagem:** Python 3.14
- **Interface:** Streamlit (Dashboard interativo estilo "L'Oréal Luxe AI Lab")
- **Machine Learning:** PyTorch, PyTorch Geometric, Scikit-Learn, Optuna
- **Química Computacional:** RDKit
- **Banco de Dados:** PostgreSQL (SQLAlchemy ORM)
- **LLM API:** Groq (Llama 3.3)

---

## 📂 Estrutura do Projeto

```text
/
├── core/                   # Cérebro da Aplicação
│   ├── chemistry.py        # Motor de física e química (volatilidade, OAV)
│   ├── compliance.py       # Regulação (IFRA) e Sustentabilidade (Eco-Score)
│   ├── discovery.py        # Orquestrador do ciclo de descoberta
│   ├── encoder.py          # Vetorização de moléculas e Grafos
│   ├── evolution.py        # Lógica do Algoritmo Genético
│   ├── market.py           # Análise financeira e fit de mercado
│   ├── model.py            # GNN (Graph Neural Network) em PyTorch
│   ├── strategy.py         # Agente LLM (Groq)
│   └── surrogate.py        # Modelo Substituto Bayesiano
├── data/
│   └── insumos.csv         # Dados brutos para seed do banco de dados
├── infra/                  # Camada de Infraestrutura
│   ├── database.py         # Conexão PostgreSQL
│   ├── models.py           # Modelos ORM (SQLAlchemy)
│   └── gemini_client.py    # Cliente API Groq
├── main.py                 # Aplicação Frontend (Streamlit)
├── migrate_db.py           # Script de inicialização do Banco de Dados
└── experiments/            # Scripts de tuning de hiperparâmetros
´´´

---

## ⚙️ Instalação e Configuração

### 1. Pré-requisitos

- Python 3.10+

- PostgreSQL instalado e rodando.

- Conta na Groq Cloud para chave de API.

### 2. Setup do Ambiente

* git clone [https://github.com/seu-usuario/perfumaria-ai.git](https://github.com/seu-usuario/perfumaria-ai.git)
* cd perfumaria-ai

# Crie o ambiente virtual
* python -m venv venv
# Windows:
* venv\Scripts\activate
# Linux/Mac:
* source venv/bin/activate

# Instale as dependências
* pip install -r requirements.txt

### 3. Configuração (.env)
* Crie um arquivo .env na raiz com as seguintes variáveis:

Snippet de código
# API Keys
GROQ_API_KEY=sua_chave_groq_aqui

# Configuração do Banco de Dados (PostgreSQL)
DB_USER=seu_usuario
DB_PASSWORD=sua_senha
DB_HOST=localhost
DB_PORT=5432
DB_NAME=perfumaria_db

```

4. Inicialização do Banco de Dados
   Antes de rodar a aplicação, é necessário migrar os dados do CSV para o PostgreSQL:

Bash
python migrate_db.py

## ▶️ Como Usar

Executando o Laboratório Virtual
Inicie a interface do Streamlit:

Bash
streamlit run main.py
Fluxo de Trabalho na Interface:
Start Synthesis: Clique para iniciar o ciclo de descoberta genética.

Dashboard: Visualize a Pirâmide Olfativa, Curva de Evaporação e Radar Sensorial.

Human-in-the-Loop: Use os sliders "Sensory Training" para dar notas (Hedônica, Técnica, Criativa) à fórmula gerada. O sistema re-treina a GNN em tempo real com seu feedback.

Green Reformulation: Se a fórmula não for sustentável, clique em "Reformulate Green" para que a IA busque substitutos biodegradáveis automaticamente.

## 🧪 Testes

O projeto conta com uma suíte de testes automatizados:

Bash
pytest tests/
