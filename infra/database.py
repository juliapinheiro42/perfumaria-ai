import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

from infra.models import Ingredient, Psychophysics

load_dotenv()


def get_db_engine():
    try:
        db_user = os.getenv('DB_USER')
        db_pass = os.getenv('DB_PASSWORD')
        db_host = os.getenv('DB_HOST')
        db_name = os.getenv('DB_NAME')
        db_port = os.getenv('DB_PORT', '5432')

        DB_URI = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        return create_engine(DB_URI)
    except Exception as e:
        print(f" Erro ao criar engine: {e}")
        return None


def load_insumos_from_db():
    """
    Carrega um dicionário otimizado para o Business Engine.
    Retorna: Dict[nome_ingrediente, {dados}]
    """
    engine = get_db_engine()
    if not engine:
        return {}

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        query = session.query(
            Ingredient.name,
            Ingredient.price,
            Ingredient.ifra_limit,
            Ingredient.traditional_use,
            Psychophysics.russell_valence,
            Psychophysics.russell_arousal
        ).outerjoin(Psychophysics, Ingredient.id == Psychophysics.ingredient_id)

        results = query.all()

        dados = {}
        for row in results:
            dados[row.name] = {
                "price_per_kg": float(row.price) if row.price else 100.0,
                "ifra_limit": float(row.ifra_limit) if row.ifra_limit else 1.0,
                "traditional_use": row.traditional_use or "",
                "russell_valence": float(row.russell_valence) if row.russell_valence else 0.0,
                "russell_arousal": float(row.russell_arousal) if row.russell_arousal else 0.0
            }

        return dados

    except Exception as e:
        print(f" Erro ao ler dados do SQL: {e}")
        return {}
    finally:
        session.close()
