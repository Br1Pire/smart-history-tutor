import random
import copy
from src.core.crossover_mutation import crossover_based_on_test_results, mutate_plan

class MetaheuristicOptimizer:
    def __init__(self, generator, evaluate_fitness, population_size=10, tournament_size=3):
        """
        Optimizer basado en Algoritmo Genético.
        
        generator: función que genera un plan aleatorio.
        evaluate_fitness: función que evalúa un plan y devuelve su fitness.
        population_size: tamaño de la población.
        tournament_size: tamaño del torneo de selección.
        """
        self.generator = generator
        self.evaluate_fitness = evaluate_fitness
        self.population_size = population_size
        self.tournament_size = tournament_size

    def initialize_population(self, subtopic_list, constraints):
        """
        Genera la población inicial.
        """
        population = []
        for _ in range(self.population_size):
            plan = self.generator(subtopic_list, constraints)
            fitness = self.evaluate_fitness(plan, constraints)
            population.append({"plan": plan, "fitness": fitness})
        return population

    def tournament_selection(self, population):
        """
        Selección por torneo: elige el mejor entre un grupo aleatorio.
        """
        tournament = random.sample(population, self.tournament_size)
        tournament.sort(key=lambda x: x["fitness"], reverse=True)
        return tournament[0]["plan"]

    def run(self, subtopic_list, constraints, num_generations=5):
        """
        Corre el algoritmo genético completo con crossover y mutación.
        """
        population = self.initialize_population(subtopic_list, constraints)

        for generation in range(num_generations):
            print(f"\n=== GENERATION {generation+1} ===")

            # Selección de padres
            parents = [self.tournament_selection(population) for _ in range(self.population_size)]

            # Crossover y mutación para generar nueva población
            new_population = []
            for i in range(0, len(parents), 2):
                parent_a = parents[i]
                parent_b = parents[i+1] if i+1 < len(parents) else parents[0]

                # Simular test_results dummy (cuando implementes simulación real, cámbialo)
                dummy_test_results_a = {s: random.uniform(0,1) for s in subtopic_list}
                dummy_test_results_b = {s: random.uniform(0,1) for s in subtopic_list}

                # Crossover basado en resultados de tests
                child_plan = crossover_based_on_test_results(
                    parent_a, parent_b,
                    dummy_test_results_a, dummy_test_results_b
                )

                # Mutación
                mutated_child = mutate_plan(child_plan, subtopic_list, constraints)

                # Evaluar fitness
                fitness = self.evaluate_fitness(mutated_child, constraints)
                new_population.append({"plan": mutated_child, "fitness": fitness})

            # Reemplazar población
            population = new_population

            # Mostrar fitness de la generación
            for ind in population:
                print(f"Fitness: {ind['fitness']:.4f}")

            # Mostrar top plan
            best = max(population, key=lambda x: x["fitness"])
            print(f"🏆 Best fitness: {best['fitness']:.4f}")

        # Retorna mejor plan encontrado
        return max(population, key=lambda x: x["fitness"])
   

def crossover_based_on_test_results(plan_a, plan_b, test_results_a, test_results_b):
    """
    Realiza crossover entre plan_a y plan_b basado en los resultados de tests.
    Sustituye clases de plan_a por las de plan_b si la calificación de un subtopic
    es mejor en plan_b.

    Args:
        plan_a (dict): Primer plan de clases.
        plan_b (dict): Segundo plan de clases.
        test_results_a (dict): {subtopic: score} para plan_a.
        test_results_b (dict): {subtopic: score} para plan_b.

    Returns:
        dict: Nuevo plan_a modificado.
    """

    new_plan = copy.deepcopy(plan_a)

    # Mapear sesiones de plan_b por subtopic para acceso rápido
    plan_b_sessions_by_subtopic = {}
    for session in plan_b["sessions"]:
        for subtopic in session["subtopics"]:
            plan_b_sessions_by_subtopic[subtopic] = session

    # Revisar cada subtopic en test_results
    for subtopic in test_results_a:
        score_a = test_results_a.get(subtopic, 0)
        score_b = test_results_b.get(subtopic, 0)

        # Si plan_b tiene mejor score y tiene la clase, sustituirla en plan_a
        if score_b > score_a and subtopic in plan_b_sessions_by_subtopic:
            for idx, session in enumerate(new_plan["sessions"]):
                if subtopic in session["subtopics"]:
                    # Reemplazar sesión completa manteniendo session_id
                    new_session = copy.deepcopy(plan_b_sessions_by_subtopic[subtopic])
                    new_session["session_id"] = session["session_id"]
                    new_plan["sessions"][idx] = new_session
                    break

    return new_plan

# ==========================================================
# MUTACIONES
# ==========================================================

def mutate_plan(plan, subtopic_list, constraints):
    """
    Realiza mutaciones simples sobre un plan:
    - Añade o elimina subtopics.
    - Reordena sesiones manteniendo prerequisitos.
    - Ajusta coverage ±10%.

    Args:
        plan (dict): Plan de clases.
        subtopic_list (list): Lista de subtemas posibles.
        constraints (dict): Restricciones generales.

    Returns:
        dict: Nuevo plan mutado.
    """
    mutated = copy.deepcopy(plan)
    taught_subtopics = {s for session in mutated["sessions"] if session["type"] == "content" for s in session["subtopics"]}
    min_classes, max_classes = constraints["num_classes_range"]

    # 1. Mutación: añadir subtopic a clase content
    available_subtopics = [s for s in subtopic_list if s not in taught_subtopics]
    content_sessions = [s for s in mutated["sessions"] if s["type"] == "content"]

    if available_subtopics and content_sessions and random.random() < 0.5:
        target_session = random.choice(content_sessions)
        if len(target_session["subtopics"]) < 2:
            new_subtopic = random.choice(available_subtopics)
            target_session["subtopics"].append(new_subtopic)
            taught_subtopics.add(new_subtopic)

    # 2. Mutación: eliminar subtopic de clase content (si coverage >= min)
    if len(taught_subtopics) > int(len(subtopic_list) * constraints["coverage"]) and random.random() < 0.3:
        removable_sessions = [s for s in content_sessions if len(s["subtopics"]) > 1]
        if removable_sessions:
            target_session = random.choice(removable_sessions)
            removed = random.choice(target_session["subtopics"])
            target_session["subtopics"].remove(removed)
            taught_subtopics.discard(removed)

    # 3. Mutación: reordenar sesiones respetando prerequisitos
    if random.random() < 0.5:
        # Separar content y review para mantener orden
        contents = [s for s in mutated["sessions"] if s["type"] == "content"]
        reviews = [s for s in mutated["sessions"] if s["type"] == "review"]
        random.shuffle(contents)
        random.shuffle(reviews)
        mutated["sessions"] = contents + reviews

    # 4. Mutación: ajustar coverage ±10% dentro de [min,1.0]
    if random.random() < 0.4:
        delta = random.uniform(-0.1, 0.1)
        new_coverage = mutated.get("coverage", constraints["coverage"]) + delta
        new_coverage = max(constraints["coverage"], min(new_coverage, 1.0))
        mutated["coverage"] = new_coverage

    return mutated


    

# =======================================================
# Ejemplo de integración

if __name__ == "__main__":
    
    from src.core.fitness import evaluate_plan_fast as evaluate_fitness
    from src.core.tutor_builder import create_tutor_instance

    tutor = create_tutor_instance()

    subtopics = ['a','b','c','d','e','f','g','h']
    subtopic_query_pairs = [(s, f"dummy query for {s}") for s in subtopics]

    constraints = {
        "coverage": 0.5,
        "num_classes_range": [2, 5]
    }

    optimizer = MetaheuristicOptimizer(tutor.generate_plan, evaluate_fitness)
    best_solution = optimizer.run(subtopic_query_pairs, constraints, num_generations=5)

    print("\n=== BEST OVERALL PLAN ===")
    for s in best_solution["plan"]["sessions"]:
        print(f" - {s['type'].upper()}: {s['subtopics'] if 'subtopics' in s else s['subtopic']}")
    print(f"✅ Fitness: {best_solution['fitness']:.4f}")
