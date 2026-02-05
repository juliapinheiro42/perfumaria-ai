import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()


def get_db_engine():
    """Cria a conexão com o banco usando variáveis de ambiente (.env)."""
    try:
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        dbname = os.getenv("DB_NAME")

        if not all([user, password, host, port, dbname]):
            # Usando st.warning para não quebrar a UI se faltar config
            print("⚠️ Configuração de Banco de Dados incompleta no .env")
            return None

        # Codifica a senha para evitar erro com caracteres especiais
        encoded_password = quote_plus(password)

        # --- MUDANÇA PRINCIPAL AQUI ---
        # Adicionamos ?client_encoding=utf8 no final da URL
        url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{dbname}?client_encoding=utf8"

        engine = create_engine(url)
        return engine
    except Exception as e:
        # Usamos repr(e) para evitar erro de decodificação na hora de printar o erro
        print(f"❌ Erro de configuração do Banco de Dados: {repr(e)}")
        return None


@st.cache_data(ttl=3600)
def load_insumos_from_db():
    """
    Carrega os dados do PostgreSQL.
    """
    engine = get_db_engine()
    if not engine:
        return {}

    query = """
    SELECT 
        i.name as name,
        m.molecular_weight,
        m.log_p as polarity,
        m.smiles,
        i.category,
        i.price as price_per_kg,
        m.description as olfactive_notes,
        i.olfactive_family,
        i.ifra_limit,
        m.vapor_pressure
    FROM ingredients i
    LEFT JOIN composition c ON i.id = c.ingredient_id
    LEFT JOIN molecules m ON c.molecule_id = m.id
    WHERE m.smiles IS NOT NULL
    """

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        if df.empty:
            print("⚠️ [DB] Query executada com sucesso, mas retornou 0 linhas.")
            return {}

        df = df.drop_duplicates(subset=['name'])
        df = df.set_index("name")

        return df.to_dict('index')

    except Exception as e:
        # MUDANÇA AQUI TAMBÉM: repr(e) evita o crash do 'ç'
        print(f"⚠️ [DB] Erro ao carregar dados do SQL: {repr(e)}")
        return {}
