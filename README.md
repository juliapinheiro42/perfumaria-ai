# 🧪✨ Perfumaria AI

### Artificial Intelligence for Molecular Design and Fragrance Optimization

> An **End-to-End platform** that simulates a Master Perfumer and a
> Sensory Evaluation Panel using Genetic Algorithms, Graph Neural
> Networks (GNNs), Bayesian Optimization, and LLMs to create innovative,
> sustainable, and commercially viable fragrances.

---

## 🌿 Vision

**Perfumaria AI** is a virtual olfactory discovery lab that combines:

- 🧬 Evolutionary formula generation\
- 🧠 Molecular modeling with GNNs\
- 📈 Intelligent chemical space optimization\
- 🌍 Automated sustainability and compliance\
- 💼 Integrated market intelligence

The platform operates with **200 Amazonian-origin ingredients**, 100%
sustainable and renewable.

---

# 🚀 Core Capabilities

## 🧠 Hybrid Intelligence Architecture

### 🔬 Genetic Algorithms

Formula evolution through: - Crossover\

- Mutation\
- Multi-objective selection

📂 `core/evolution.py`

---

### 🧠 Graph Neural Network (GNN)

Built with **PyTorch Geometric**, the model: - Represents molecules as
graphs\

- Learns structural patterns\
- Predicts olfactory performance\
- Continuously adapts via human feedback

📂 `core/model.py`

---

### 🤖 LLM Strategic Agent (Groq + Llama 3)

A cognitive agent that: - Analyzes evolutionary history\

- Identifies successful patterns\
- Suggests high-level strategy shifts (e.g., increase top-note
  volatility)

📂 `core/strategy.py`

---

### 📊 Bayesian Optimization

Surrogate modeling using **Gaussian Processes** to: - Efficiently
explore chemical space\

- Reduce redundant experimentation\
- Maximize multi-objective performance

📂 `core/surrogate.py`

---

# ⚗️ Physicochemical Simulation

📂 `core/chemistry.py`

### 🌫 4D Evaporation Curve

Simulates volatility from 0h to 10h: - Fragrance evolution\

- Pyramid transition over time

### 🌬 Projection (Sillage)

Estimated from: - Vapor pressure\

- Odor Activity Value (OAV)

### 🧠 Neuro-Impact (Russell Model)

Maps formula into: - Valence\

- Arousal

Predicts emotional states such as relaxation, energy, and
sophistication.

---

# 🌍 Sustainability & Compliance

📂 `core/compliance.py`

### ✔ IFRA Verification

- Automatic regulatory limit checking

### ♻ Dynamic Eco-Score

Calculates: - Carbon footprint\

- Biodegradability\
- Renewability

### 🌱 Green Reformulation

Automatically replaces non-sustainable ingredients with biodegradable
alternatives while preserving the olfactory profile.

Function: `reformulate_green()`

---

# 💼 Market Intelligence

📂 `core/market.py`

### 🌎 Regional Fit Analysis

Cultural suitability assessment for: - Asia\

- Latin America\
- Middle East\
- United States

### 💰 Dynamic Pricing

- Juice cost estimation\
- Gross margin projection\
- Suggested tier: Mass \| Prestige \| Luxury

---

# 🛠 Tech Stack

Layer Technology

---

Language Python 3.10+
Interface Streamlit
ML PyTorch, PyTorch Geometric, Scikit-Learn, Optuna
Computational Chemistry RDKit
Database PostgreSQL + SQLAlchemy
LLM Groq (Llama 3.x)

---

# 📂 Project Structure

```text
/
├── core/
│   ├── chemistry.py
│   ├── compliance.py
│   ├── discovery.py
│   ├── encoder.py
│   ├── evolution.py
│   ├── market.py
│   ├── model.py
│   ├── strategy.py
│   └── surrogate.py
│
├── data/
│   └── insumos.csv
│
├── infra/
│   ├── database.py
│   ├── models.py
│   └── gemini_client.py
│
├── experiments/
├── tests/
├── migrate_db.py
└── main.py
```

---

# ⚙️ Installation

## 1. Requirements

- Python 3.10+
- PostgreSQL
- Groq Cloud account

---

## 2. Setup

```bash
git clone https://github.com/your-username/perfumaria-ai.git
cd perfumaria-ai

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

---

## 3. Environment Configuration

Create a `.env` file in the project root:

```env
# API
GROQ_API_KEY=your_groq_key_here

# Database
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=perfumaria_db
```

---

## 4. Initialize Database

```bash
python migrate_db.py
```

---

# ▶️ Running the Application

```bash
streamlit run main.py
```

---

# 🖥 Usage Flow

### 🔹 Start Synthesis

Launches the evolutionary discovery cycle.

### 🔹 Dashboard

Visualize: - Olfactory Pyramid\

- Evaporation Curve\
- Sensory Radar\
- Projection & Longevity

### 🔹 Human-in-the-Loop

Evaluate generated formulas via sliders: - Hedonic\

- Technical\
- Creative

The GNN is retrained dynamically based on feedback.

### 🔹 Green Reformulation

Automatically reformulates non-sustainable compositions.

---

# 🧪 Testing

```bash
pytest tests/
```

---

# 🎯 Mission

**Perfumaria AI** merges science, art, and artificial intelligence to
revolutionize fragrance creation through sustainable molecular
innovation.
