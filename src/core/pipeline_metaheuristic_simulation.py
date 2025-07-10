import logging
import random
import json
from src.core.metaheuristics import MetaheuristicOptimizer
from src.core.simulation import run_simulation_cycle
from src.core.tutor_builder import create_tutor_instance
from src.core.student_builder import create_multiple_students
from src import config
from src import params

# ====== Configuración de logging ======
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def run_pipeline(constraints = params.CONSTRAINTS, topic = params.TOPIC, topics_rangec = params.TOPICS_RANGE, population_size = params.POPULATION_SIZE, tournament_size = params.TOURNAMET_SIZE, students_amount = params.STUDENTS_AMOUNT, students_proportions = params.STUDENTS_PROPORTION):
    logging.info("Iniciando pipeline de simulación.")

    optimizer = MetaheuristicOptimizer(population_size, tournament_size)

    logging.info("Creando instancia de tutor.")
    tutor = create_tutor_instance()

    logging.info(f"Creando {students_amount} estudiantes con proporciones {students_proportions}.")
    students = create_multiple_students(students_amount, students_proportions)

    sim_path = config.DATA_DIR / 'simulation'  
    gen_path = sim_path / "generations"

    logging.info(f"Generando subtemas para el tópico: {topic}.")
    subtopic_query_pairs = tutor.generate_subtopics(topic, topics_range)
    subtopics = [x[0] for x in subtopic_query_pairs]

    logging.info("Generando textos a partir de los subtemas.")
    texts = tutor.generate_texts_from_subtopics(tutor.retrive_chunks_from_subtopics(subtopic_query_pairs))
    logging.info("Generando tests a partir de los textos.")
    tests = tutor.generate_tests_from_subtopics(texts)

    with open(sim_path / "texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
        logging.info("Texts guardados en texts.json.")

    with open(sim_path / "tests.json", "w", encoding="utf-8") as f:
        json.dump(tests, f, ensure_ascii=False, indent=2)
        logging.info("Tests guardados en tests.json.")

    logging.info("Ejecutando generación cero.")
    parents = optimizer.run_generation_zero(subtopic_query_pairs, constraints)

    evaluated_parents = []
    for idx, p in enumerate(parents):
        logging.info(f"Evaluando padre inicial {idx+1}/{len(parents)}.")
        enriched_plan = tutor.generate_plan(p, texts, tests, constraints)
        simulation_result = run_simulation_cycle(enriched_plan, tutor, students)
        fitness = optimizer.evaluate_fitness_with_scores(p, simulation_result['global_average_score'], constraints)
        evaluated_parents.append({
            "distribution": p,
            "fitness": fitness
        })

    with open(gen_path / f"GENERATION 0.json", "w", encoding="utf-8") as f:
        json.dump(evaluated_parents, f, ensure_ascii=False, indent=2)
        logging.info("Generación 0 guardada.")

    NUM_GENERATIONS = params.NUM_GENERATIONS
    elitism_size = max(1, optimizer.population_size // 10)

    for generation in range(NUM_GENERATIONS):
        logging.info(f"=== GENERACIÓN {generation+1} ===")

        evaluated_parents.sort(key=lambda x: x["fitness"], reverse=True)
        elite = evaluated_parents[:elitism_size]
        logging.info(f"Elite seleccionada (top {elitism_size}).")

        children = []
        while len(children) < optimizer.population_size - elitism_size:
            p1 = random.choice([x["distribution"] for x in evaluated_parents])
            p2 = random.choice([x["distribution"] for x in evaluated_parents])
            child = optimizer.crossover(p1, p2)
            child = optimizer.mutate(child, subtopics)
            children.append(child)
        logging.info(f"{len(children)} hijos generados por crossover y mutación.")

        evaluated_children = []
        for idx, child in enumerate(children):
            logging.info(f"Evaluando hijo {idx+1}/{len(children)}.")
            enriched_plan = tutor.generate_plan(child, texts, tests, constraints)
            simulation_result = run_simulation_cycle(enriched_plan, tutor, students)
            fitness = optimizer.evaluate_fitness_with_scores(child, simulation_result['global_average_score'], constraints)
            evaluated_children.append({
                "distribution": child,
                "fitness": fitness
            })

        combined_population = elite + evaluated_children
        combined_population.sort(key=lambda x: x["fitness"], reverse=True)
        evaluated_parents = combined_population[:optimizer.population_size]

        best = evaluated_parents[0]

        save = {
            'best': best,
            'distributions': evaluated_parents
        }
        with open(gen_path / f"GENERATION {generation+1}.json", "w", encoding="utf-8") as f:
            json.dump(save, f, ensure_ascii=False, indent=2)
            logging.info(f"Generación {generation+1} guardada.")

        logging.info(f"🏆 Mejor fitness (generación {generation+1}): {best['fitness']:.4f}")

    logging.info("=== RESULTADOS FINALES ===")
    final_best = evaluated_parents[0]
    logging.info(f"✅ Mejor plan encontrado con fitness: {final_best['fitness']:.4f}")
    return final_best

if __name__ == '__main__':

    constraints = {
        "coverage": 0.5,
        "num_classes_range": [5, 5]
    }
    topic = "Historia Contemporanea"
    topics_range = (3, 5)
    population_size = 6
    tournament_size = 3
    students_amount = 3
    students_proportions = (100,0,0)

    run_pipeline(constraints, topic, topics_range, population_size, tournament_size, students_amount, students_proportions)
