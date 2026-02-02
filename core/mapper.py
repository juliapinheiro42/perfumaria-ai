import pandas as pd
from difflib import get_close_matches

class NoteMapper:
    def __init__(self, csv_path):
        try:
            self.df = pd.read_csv(csv_path)
            self.df.columns = self.df.columns.str.strip()
            # Cria um dicionário de busca rápida
            self.inventory = self.df['name'].unique().tolist()
            # Dicionário reverso para buscar por notas olfativas
            self.note_to_mol = {}
            for _, row in self.df.iterrows():
                # Assume que existe uma coluna 'olfactive_notes' ou similar
                notes = str(row.get('olfactive_notes', '')).lower()
                mol_name = row['name']
                for note in notes.split():
                    clean_note = note.strip(',. ')
                    if clean_note not in self.note_to_mol:
                        self.note_to_mol[clean_note] = []
                    self.note_to_mol[clean_note].append(mol_name)
                    
        except Exception as e:
            print(f"[MAPPER ERROR] {e}")
            self.inventory = []

    def get_best_match(self, note_name):
        """Encontra a melhor molécula para uma nota olfativa (ex: 'Pear' -> 'Hexyl Acetate')"""
        note = note_name.lower().strip()
        
        # 1. Dicionário de Tradução Direta (O Segredo do Idôle)
        # Mapeia notas comerciais para químicos comuns
        mappings = {
            # FRUTAS
            'pear': ['Hexyl Acetate', 'Ethyl Decadienoate', 'Geranyl Acetate'], # Evita Allyl Caproate (Abacaxi)
            'pineapple': ['Allyl Caproate', 'Allyl Amyl Glycolate', 'Pharaone'],
            'apple': ['Verdox', 'Manzanate', 'Fructone'],
            'bergamot': ['Linalyl Acetate', 'Limonene', 'Bergamot Oil'],
            
            # FLORES
            'rose': ['Geraniol', 'Citronellol', 'Phenyl Ethyl Alcohol', 'Geranyl Acetate', 'Rose Oxide'],
            'jasmine': ['Hedione', 'Benzyl Acetate', 'Indole', 'Hexyl Cinnamic Aldehyde'],
            'lily': ['Florol', 'Lilial', 'Lyral'],
            
            # FUNDO
            'musk': ['Galaxolide', 'Habanolide', 'Musk Ketone', 'Ethylene Brassylate', 'Ambrettolide'],
            'vanilla': ['Vanillin', 'Ethyl Vanillin', 'Coumarin'],
            'cedar': ['Iso E Super', 'Vertofix', 'Cedrol'],
            'sandalwood': ['Ebanol', 'Javanol', 'Bacdanol', 'Sandalore'],
            'amber': ['Ambroxan', 'Amberwood', 'Timberol']
        }
        
        # Tenta encontrar no mapeamento manual
        if note in mappings:
            candidates = mappings[note]
            # Verifica quais candidatos existem no nosso estoque real
            valid_candidates = [m for m in candidates if m in self.inventory]
            if valid_candidates:
                return valid_candidates[0] # Retorna a melhor opção disponível
        
        # 2. Busca por Palavra-Chave nas Notas do CSV
        if note in self.note_to_mol:
            return self.note_to_mol[note][0]
            
        # 3. Busca Aproximada (Fuzzy)
        matches = get_close_matches(note, self.inventory, n=1, cutoff=0.6)
        if matches:
            return matches[0]
            
        return None

def process_target_perfume(target_data, mapper):
    """Converte o JSON do perfume alvo em lista de moléculas"""
    molecules = []
    
    # Extrai todas as notas do texto (Top, Middle, Base)
    all_notes = []
    for part in ['Top', 'Middle', 'Base']:
        if part in target_data and target_data[part]:
            # Separa por vírgula e limpa espaços
            notes = [n.strip() for n in str(target_data[part]).split(',')]
            all_notes.extend(notes)
            
    print(f"[MAPPER] Traduzindo notas: {all_notes}")
    
    for note in all_notes:
        match = mapper.get_best_match(note)
        if match:
            # Recupera dados completos da molécula do DataFrame original
            row = mapper.df[mapper.df['name'] == match].iloc[0]
            mol_data = {
                "name": row['name'],
                "smiles": row['smiles'],
                "category": row['category'],
                "price_per_kg": row['price_per_kg'],
                "molecular_weight": row.get('molecular_weight', 200)
            }
            molecules.append(mol_data)
        else:
            print(f"   [AVISO] Sem correspondência para nota: '{note}'")
            
    return molecules