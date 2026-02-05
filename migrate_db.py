import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

from infra.models import (
    create_all_tables, Molecule, Ingredient, Composition,
    Psychophysics, Sustainability
)

load_dotenv()

DB_URI = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)


def clean_percent(value):
    """Utilitário para limpar porcentagens."""
    if pd.isna(value) or value == '':
        return None
    if isinstance(value, str):
        value = value.replace('%', '').strip()
        if value.lower() == 'restricted':
            return 0.01
    try:
        val = float(value)
        return val / 100.0 if val > 1.0 else val
    except ValueError:
        return None


def migrate_insumos():
    create_all_tables(engine)

    session = Session()
    try:
        print(" Lendo CSV...")
        df = pd.read_csv("data/insumos.csv")
        df.columns = df.columns.str.strip().str.lower()

        print(f" Iniciando migração de {len(df)} itens via ORM...")

        for _, row in df.iterrows():
            smiles = row.get('smiles')
            if pd.isna(smiles):
                continue

            molecule = session.query(Molecule).filter_by(smiles=smiles).first()
            if not molecule:
                molecule = Molecule(
                    smiles=smiles,
                    molecular_weight=row.get('molecular_weight'),
                    log_p=row.get('polarity')
                )
                session.add(molecule)
                session.flush()

            ingredient_name = row['name']
            ingredient = session.query(Ingredient).filter_by(
                name=ingredient_name).first()

            ifra_val = clean_percent(row.get('ifra_limit'))
            is_allergen_val = str(row.get('is_allergen')).lower() == 'true'

            if not ingredient:
                ingredient = Ingredient(
                    name=ingredient_name,
                    category=row.get('category'),
                    olfactive_family=row.get('olfactive_family'),
                    olfactive_notes=row.get('olfactive_notes'),
                    price=row.get('price_per_kg'),
                    ifra_limit=ifra_val,
                    is_allergen=is_allergen_val,
                    complexity_tier=row.get('complexity_tier'),
                    traditional_use=row.get('traditional_use')
                )
                session.add(ingredient)
                session.flush()

            comp_exists = session.query(Composition).filter_by(
                ingredient_id=ingredient.id,
                molecule_id=molecule.id
            ).first()

            if not comp_exists:
                comp = Composition(
                    ingredient=ingredient,
                    molecule=molecule,
                    quantity=1.0
                )
                session.add(comp)

            if not ingredient.psychophysics:
                psych = Psychophysics(
                    ingredient=ingredient,
                    odor_threshold_ppb=row.get('odor_threshold_ppb'),
                    odor_potency=row.get('odor_potency'),
                    russell_valence=row.get('russell_valence'),
                    russell_arousal=row.get('russell_arousal'),
                    evidence_level=row.get('evidence_level')
                )
                session.add(psych)

            if not ingredient.sustainability:
                sust = Sustainability(
                    ingredient=ingredient,
                    biodegradability=str(
                        row.get('biodegradability')).lower() == 'true',
                    renewable_source=str(
                        row.get('renewable_source')).lower() == 'true',
                    carbon_footprint=row.get('carbon_footprint')
                )
                session.add(sust)

        session.commit()
        print(" Migração concluída com sucesso!")

    except Exception as e:
        session.rollback()
        print(f" Erro na migração: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    migrate_insumos()
