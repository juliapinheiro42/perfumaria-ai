import pandas as pd
import numpy as np
from sqlalchemy import text
from infra.database import get_db_engine


def seed_database():
    print(" Iniciando população do Banco de Dados...")

    try:
        df = pd.read_csv("insumos.csv")
        df = df.replace({np.nan: None})
        print(f" CSV carregado: {len(df)} insumos encontrados.")
    except FileNotFoundError:
        print(" Erro: Arquivo 'insumos.csv' não encontrado.")
        return

    # 2. Conectar ao Banco
    engine = get_db_engine()
    if not engine:
        print(" Erro: Sem conexão com o banco.")
        return

    sucesso = 0
    erros = 0

    with engine.connect() as conn:
        for index, row in df.iterrows():

            try:
                with conn.begin():

                    check_mol = text(
                        "SELECT id FROM molecules WHERE smiles = :smiles")
                    smiles_val = row['smiles'] if row['smiles'] else f"unknown_{index}"

                    res_mol = conn.execute(
                        check_mol, {"smiles": smiles_val}).fetchone()

                    if res_mol:
                        mol_id = res_mol[0]
                    else:
                        insert_mol = text("""
                            INSERT INTO molecules (name, molecular_weight, description, smiles, log_p, vapor_pressure)
                            VALUES (:name, :mw, :desc, :smiles, :log_p, :vp)
                            RETURNING id
                        """)
                        mol_id = conn.execute(insert_mol, {
                            "name": row['name'],
                            "mw": row['molecular_weight'] if row['molecular_weight'] else 0.0,
                            "desc": row['olfactive_notes'],
                            "smiles": smiles_val,
                            "log_p": row['polarity'] if row['polarity'] else 0.0,
                            "vp": 0.01
                        }).fetchone()[0]

                    ifra_raw = str(row['ifra_limit']).replace(
                        '%', '').replace('<', '').strip()
                    try:
                        ifra_val = float(ifra_raw)
                    except:
                        ifra_val = 100.0

                    check_ing = text(
                        "SELECT id FROM ingredients WHERE name = :name")
                    res_ing = conn.execute(
                        check_ing, {"name": row['name']}).fetchone()

                    if res_ing:
                        ing_id = res_ing[0]
                    else:
                        insert_ing = text("""
                            INSERT INTO ingredients (name, category, price, olfactive_family, ifra_limit)
                            VALUES (:name, :cat, :price, :family, :ifra)
                            RETURNING id
                        """)
                        ing_id = conn.execute(insert_ing, {
                            "name": row['name'],
                            "cat": row['category'],
                            "price": row['price_per_kg'] if row['price_per_kg'] else 0.0,
                            "family": row['olfactive_family'],
                            "ifra": ifra_val
                        }).fetchone()[0]

                    check_comp = text(
                        "SELECT * FROM composition WHERE ingredient_id=:ing AND molecule_id=:mol")
                    res_comp = conn.execute(
                        check_comp, {"ing": ing_id, "mol": mol_id}).fetchone()

                    if not res_comp:
                        insert_comp = text("""
                            INSERT INTO composition (ingredient_id, molecule_id, percentage)
                            VALUES (:ing_id, :mol_id, 100.0)
                        """)
                        conn.execute(
                            insert_comp, {"ing_id": ing_id, "mol_id": mol_id})

                    sucesso += 1

            except Exception as e:
                print(f" Erro na linha {index} ({row['name']}): {e}")
                erros += 1

    print("-" * 30)
    print(
        f" {sucesso} inseridos com sucesso | {erros} falhas.")


if __name__ == "__main__":
    seed_database()
