
# ==============================================================================
# BIBLIOTECA DE ACORDES (BLOCOS DE CONSTRUÇÃO)
# ==============================================================================
ACORDES_LIB = {
    "Grojsman_Accord": {
        "molecules": ["Iso E Super", "Hedione", "Galaxolide", "Methyl Ionone"], 
        "ratios": [0.25, 0.25, 0.25, 0.25], 
        "description": "O famoso acorde 'abraço' que serve de base para 80% dos perfumes modernos."
    },
    "Amber_Base": {
        "molecules": ["Vanillin", "Labdanum Resinoid", "Ambroxan"],
        "ratios": [0.4, 0.3, 0.3],
        "description": "Base oriental clássica."
    },
    "Fougere_Core": {
        "molecules": ["Lavender Oil", "Coumarin", "Oakmoss Absolute", "Geraniol"],
        "ratios": [0.4, 0.2, 0.2, 0.2],
        "description": "Estrutura masculina clássica (Barbearia)."
    },
    "Clean_Musk_Accord": {
        "molecules": ["Galaxolide", "Ethylene Brassylate", "Ambrettolide"],
        "ratios": [0.5, 0.3, 0.2],
        "description": "Cheiro de pele limpa e amaciante."
    },

    "Rose_Accord": {
        "molecules": ["Phenyl Ethyl Alcohol", "Citronellol", "Geraniol", "Rose Oxide"],
        "ratios": [0.5, 0.3, 0.15, 0.05],
        "description": "Reconstrução de rosa genérica."
    },
    "White_Floral_Accord": {
        "molecules": ["Benzyl Salicylate", "Hedione", "Indole", "Methyl Anthranilate"],
        "ratios": [0.5, 0.4, 0.05, 0.05],
        "description": "Jasmim/Tuberosa solar."
    },

    "Citrus_Top": {
        "molecules": ["Limonene", "Bergamot Oil", "Citral", "Mandarin Oil"],
        "ratios": [0.4, 0.3, 0.1, 0.2],
        "description": "Saída cítrica explosiva."
    },
    "Ambroxan_Overdose": {
        "molecules": ["Ambroxan"],
        "ratios": [1.0],
        "description": "Dose maciça de ambroxan moderno."
    },
    "Iso_E_Super_Boost": {
        "molecules": ["Iso E Super"],
        "ratios": [1.0],
        "description": "Aveludado amadeirado puro."
    }
}

# ==============================================================================
# ESQUELETOS (ARQUITETURAS COMPLETAS)
# ==============================================================================
PERFUME_SKELETONS = {
    "Modern_Floral": ["Grojsman_Accord", "Rose_Accord", "Clean_Musk_Accord"],
    "Classic_Oriental": ["Amber_Base", "Rose_Accord", "Citrus_Top"],
    "Neo_Fougere": ["Fougere_Core", "Clean_Musk_Accord", "Ambroxan_Overdose"],
    "Skin_Scent": ["Clean_Musk_Accord", "Iso_E_Super_Boost"]
}