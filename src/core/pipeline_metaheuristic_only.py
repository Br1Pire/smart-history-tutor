import logging
import random
import json
from src.core.metaheuristics import MetaheuristicOptimizer
from src.core.tutor_builder import create_tutor_instance
from src import config

# ====== Configuración de logging ======
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def run_pipeline(constraints, topic, topics_range, population_size, tournament_size):
    optimizer = MetaheuristicOptimizer(population_size=population_size, tournament_size=tournament_size)
    tutor = create_tutor_instance()

    subtopic_query_pairs = tutor.generate_subtopics(topic, topics_range)
    subtopics = [x[0] for x in subtopic_query_pairs]

    path = config.DATA_DIR / "simulation" / "generations"

    parents = optimizer.run_generation_zero(subtopic_query_pairs, constraints)

    evaluated_parents = []
    for p in parents:
        fake_score = random.uniform(0.4, 1.0)
        fitness = optimizer.evaluate_fitness_with_scores(p["distribution"], fake_score, constraints)
        evaluated_parents.append({
            "distribution": p["distribution"],
            "fitness": fitness
        })

    with open(path / f"GENERATION 0.json", "w", encoding="utf-8") as f:
        json.dump(evaluated_parents, f, ensure_ascii=False, indent=2)

    NUM_GENERATIONS = 100
    ELITISM_RATE = 0.1

    for generation in range(NUM_GENERATIONS):
        logging.info(f"=== GENERATION {generation+1} ===")

        elite_size = max(1, int(ELITISM_RATE * optimizer.population_size))
        elite = sorted(evaluated_parents, key=lambda x: x["fitness"], reverse=True)[:elite_size]
        logging.info(f"🔝 Elitism: {elite_size} mejores individuos preservados.")

        children = []
        while len(children) < (optimizer.population_size - elite_size):
            p1 = optimizer.tournament_selection(evaluated_parents)["distribution"]
            p2 = optimizer.tournament_selection(evaluated_parents)["distribution"]
            child = optimizer.crossover(p1, p2)
            child = optimizer.mutate(child, subtopics)
            children.append(child)

        evaluated_children = []
        for child in children:
            fake_score = random.uniform(0.4, 1.0)
            fitness = optimizer.evaluate_fitness_with_scores(child, fake_score, constraints)
            evaluated_children.append({
                "distribution": child,
                "fitness": fitness
            })

        combined = elite + evaluated_children
        evaluated_parents = sorted(combined, key=lambda x: x["fitness"], reverse=True)[:optimizer.population_size]

        best = evaluated_parents[0]

        save = {
            'best': best,
            'distributions': evaluated_parents
        }

        with open(path / f"GENERATION {generation+1}.json", "w", encoding="utf-8") as f:
            json.dump(save, f, ensure_ascii=False, indent=2)

        logging.info(f"🏆 Mejor fitness (generación {generation+1}): {best['fitness']:.4f}")

    final_best = evaluated_parents[0]
    logging.info("=== FINAL RESULTS ===")
    logging.info(f"✅ Mejor plan encontrado con fitness: {final_best['fitness']:.4f}")

if __name__ == '__main__':

    constraints = {
        "coverage": 0.5,
        "num_classes_range": [10, 15]
    }
    topic = "Historia Contemporanea"
    topics_range = (10, 15)
    population_size = 100
    tournament_size = 10

    run_pipeline(constraints, topic, topics_range, population_size, tournament_size)
