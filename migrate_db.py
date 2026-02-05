import pandas as pd
from sqlalchemy import create_engine
import json

DB_URI = "postgresql://postgres:senha123260298@localhost:5432/perfumaria"
engine = create_engine(DB_URI)


def migrate():
    print("⏳ Lendo CSV...")
    df = pd.read_csv("insumos.csv")

    df.columns = df.columns.str.strip().str.lower()

    df_sql = pd.DataFrame()
    df_sql['name'] = df['name']
    df_sql['olfactive_family'] = df['olfactive_family']
    df_sql['price_per_kg'] = pd.to_numeric(
        df['price_per_kg'], errors='coerce').fillna(0)
    df_sql['category'] = df['category']

    print(f"🚀 Migrando {len(df_sql)} ingredientes para o PostgreSQL...")

    try:
        df_sql.to_sql('ingredients', engine, if_exists='append', index=False)
        print("✅ Sucesso! Dados migrados.")
    except Exception as e:
        print(f"❌ Erro na migração: {e}")


if __name__ == "__main__":
    migrate()
