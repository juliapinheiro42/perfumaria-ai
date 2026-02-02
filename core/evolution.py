import random
import copy
import numpy as np

class EvolutionEngine:
    def __init__(
        self,
        discovery_engine,
        population_size=40,
        elite_ratio=0.2,
        mutation_rate=0.2,
        mutation_strength=0.1,
        random_injection=0.1,
    ):
        self.discovery = discovery_engine
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.random_injection = random_injection

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            molecules = self.discovery._generate_molecules(strategy=None)
            population.append({"molecules": molecules})
        return population

    def evolve(self, generations=10):
        population = self.initialize_population()

        for gen in range(generations):
            print(f"\n GENERATION {gen}\n")

            evaluated = self.evaluate_population(population)
            elite = self.select_elite(evaluated)

            population = self.next_generation(elite)

        return evaluated

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    def evaluate_population(self, population):
        evaluated = []

        for individual in population:
            result = self.discovery.evaluate(individual["molecules"])
            if result is not None:
                evaluated.append(result)

        return evaluated

    # --------------------------------------------------
    # SELECTION
    # --------------------------------------------------

    def select_elite(self, evaluated):
        ranked = sorted(
            [r for r in evaluated if r.get("market") != "Invalid"],
            key=lambda r: r["fitness"],
            reverse=True
        )

        if not ranked:
            return []

        k = max(1, int(len(ranked) * self.elite_ratio))
        elite = ranked[:k]

        print(f"Elite selected: {len(elite)} / {len(evaluated)}")
        return elite

    # --------------------------------------------------
    # REPRODUCTION
    # --------------------------------------------------

    def next_generation(self, elite):
        new_population = []

        if len(elite) == 0:
            return self.initialize_population()

        if len(elite) == 1:
            elite = elite * 2

        for e in elite:
            new_population.append({
                "molecules": copy.deepcopy(e["molecules"])
            })

        num_random = int(self.population_size * self.random_injection)
        for _ in range(num_random):
            molecules = self.discovery._generate_molecules(strategy=None)
            new_population.append({"molecules": molecules})

        while len(new_population) < self.population_size:
            parent_a, parent_b = random.sample(elite, 2)

            child = self.crossover(parent_a, parent_b)
            child = self.mutate(child)

            new_population.append({"molecules": child})

        return new_population

    # --------------------------------------------------
    # GENETIC OPERATORS
    # --------------------------------------------------

    def crossover(self, parent_a, parent_b):
        a = parent_a["molecules"]
        b = parent_b["molecules"]

        # Define pontos de corte iniciais
        cut_a = random.randint(1, len(a))
        cut_b = random.randint(0, len(b) - 1)

        # Ajusta os pontos de corte para não quebrar acordes (Proteção de Blocos)
        cut_a = self._adjust_cut_point(a, cut_a)
        cut_b = self._adjust_cut_point(b, cut_b)

        # Une os segmentos dos progenitores
        child = copy.deepcopy(a[:cut_a] + b[cut_b:])

        # Remove duplicatas mantendo a integridade e ordem
        seen = set()
        filtered = []
        for m in child:
            name = m.get("name")
            if name not in seen:
                filtered.append(m)
                seen.add(name)

        child = filtered

        # Garante limites viáveis de tamanho de fórmula
        if len(child) < 2:
            child = copy.deepcopy(a)
        if len(child) > 6:
            child = child[:6]

        return child

    def _adjust_cut_point(self, molecules, cut_point):
        """
        Verifica se o ponto de corte incide sobre uma molécula de um acorde.
        Se sim, desloca o corte para manter o bloco unido.
        """
        if cut_point <= 0 or cut_point >= len(molecules):
            return cut_point

        target_mol = molecules[cut_point - 1]
        accord_id = target_mol.get("accord_id")

        if accord_id:
            # Procura o fim do bloco do acorde para mover o corte
            last_index = cut_point
            for i in range(cut_point, len(molecules)):
                if molecules[i].get("accord_id") == accord_id:
                    last_index = i + 1
                else:
                    break
            return last_index
        
        return cut_point

    def mutate(self, molecules):
        for i, m in enumerate(molecules):
            if random.random() < self.mutation_rate:
                
                # Reduz a chance de mutação se a molécula for parte de um acorde estável
                if m.get("accord_id") and random.random() < 0.5:
                    continue

                if random.random() < 0.3:
                    molecules[i] = self.discovery._random_molecule()
                    continue

                # Mutação de propriedades físico-químicas
                mw = m.get("molecular_weight", 150.0)
                mw *= random.uniform(1 - self.mutation_strength, 1 + self.mutation_strength)
                m["molecular_weight"] = float(np.clip(mw, 80, 600))

                pol = m.get("polarity", 2.0)
                pol *= random.uniform(1 - self.mutation_strength, 1 + self.mutation_strength)
                m["polarity"] = float(np.clip(pol, 0.1, 6.0))

                bp = m.get("boiling_point", 200.0)
                bp *= random.uniform(1 - self.mutation_strength, 1 + self.mutation_strength)
                m["boiling_point"] = float(np.clip(bp, 80, 450))

        return molecules