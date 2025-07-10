import random
import logging
from src.config import LOG_FILES

LOG_FILE = LOG_FILES["metaheuristic"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
class MetaheuristicOptimizer:
    def __init__(self, population_size=10, tournament_size=3):
        self.population_size = population_size
        self.tournament_size = tournament_size

    def initialize_population(self, subtopic_query_pairs, constraints):
        logging.info("🚀 Inicializando población")
        population = []
        for i in range(self.population_size):
            dist = self.generate_class_distribution(subtopic_query_pairs, constraints)
            fitness = self.evaluate_class_distribution_fast(dist, constraints)
            population.append({"distribution": dist, "fitness": fitness})
            logging.info(f"🔢 Individuo {i+1} -> Clases: {dist['num_classes']}, Cobertura: {dist['coverage']:.2f}, Fitness: {fitness:.4f}")
        return population
    
    def run_generation_zero(self, subtopic_list, constraints):
        """
        Corre solo la generación inicial y selección de padres (sin crossover ni mutation).
        """
        population = self.initialize_population(subtopic_list, constraints)

        print("\n=== GENERACIÓN 0 ===")
        for ind in population:
            print(f"Fitness rápido: {ind['fitness']:.4f}")

        parents = [self.tournament_selection(population) for _ in range(self.population_size // 2)]

        print("\n=== Padres seleccionados ===")
        for p in parents:
            print(f"Num clases: {p['num_classes']}, Cobertura: {p['coverage']:.2f}")

        return parents

    def generate_class_distribution(self, subtopic_query_pairs, constraints):
        """
        Genera un plan de clases heurístico con target coverage aleatorio,
        mezclando clases de contenido y repaso desde el inicio.
        """
        subtopic_list = [pair[0] for pair in subtopic_query_pairs]

        min_coverage = constraints["coverage"]
        min_classes, max_classes = constraints["num_classes_range"]
        num_classes = random.randint(min_classes, max_classes)

        plan = []
        taught_subtopics = set()
        review_counts = {}

        target_coverage = random.uniform(min_coverage, 1.0)

        subtopics_shuffled = subtopic_list.copy()
        random.shuffle(subtopics_shuffled)

        class_id = 1

        while len(plan) < num_classes:
            current_coverage = len(taught_subtopics) / len(subtopic_list)

            if current_coverage < target_coverage and (random.random() < 0.7 or len(taught_subtopics) == 0):
                available_content = [s for s in subtopics_shuffled if s not in taught_subtopics]
                if available_content:
                    num_subtopics = random.randint(1, min(2, len(available_content)))
                    subtopics_for_class = random.sample(available_content, num_subtopics)

                    plan.append({
                        "session_id": class_id,
                        "type": "content",
                        "subtopics": subtopics_for_class
                    })
                    class_id += 1

                    taught_subtopics.update(subtopics_for_class)
                    for s in subtopics_for_class:
                        review_counts.setdefault(s, 0)
                else:
                    pass
            else:
                if random.random() < 0.5:
                    available_review = [s for s in taught_subtopics if review_counts.get(s, 0) < 1]
                    if available_review:
                        num_subtopics = random.randint(1, min(3, len(available_review)))
                        subtopics_for_review = random.sample(available_review, num_subtopics)

                        plan.append({
                            "session_id": class_id,
                            "type": "review",
                            "subtopics": subtopics_for_review
                        })
                        class_id += 1

                        for s in subtopics_for_review:
                            review_counts[s] += 1
                    else:
                        continue
                else:
                    if current_coverage < target_coverage:
                        continue
                    else:
                        break

        coverage = len(taught_subtopics) / len(subtopic_list)

        print(f"✅ Distribucion de clases generada con {len(plan)} clases. Covertura objetivo: {target_coverage:.2%}. Cobertura alcanzada: {coverage:.2%}")
        for s in plan:
            print(f"📝 {s}")

        return {
            "num_classes": len(plan),
            "coverage": coverage,
            "sessions": plan
        }

    def evaluate_class_distribution_fast(self, distribution, constraints):
        coverage = distribution["coverage"]
        num_classes = distribution["num_classes"]
        sessions = distribution["sessions"]

        min_coverage = constraints["coverage"]
        min_classes, max_classes = constraints["num_classes_range"]
        ideal_classes = (min_classes + max_classes) / 2

        taught_subtopics = set()
        valid_plan = True

        for session in distribution["sessions"]:
            if session["type"] == "content":
                taught_subtopics.update(session["subtopics"])
            elif session["type"] == "review":
                for subtopic in session["subtopics"]:
                    if subtopic not in taught_subtopics:
                        valid_plan = False
                        logging.warning(f"❌ Plan inválido: repaso de '{subtopic}' antes de contenido.")
                        break
            if not valid_plan:
                break

        if not valid_plan:
            logging.info("⚠️ Plan inválido detectado. Fitness = 0.0")
            return 0.0

        coverage_score = max(0.0, min((coverage - min_coverage) / (1 - min_coverage), 1.0)) if coverage >= min_coverage else 0.0

        num_reviews = sum(1 for s in sessions if s["type"] == "review")
        total = len(sessions)
        review_ratio = num_reviews / total if total else 0.0
        balance_score = 1.0 - (review_ratio / 0.4) * 0.2 if review_ratio <= 0.4 else max(0.0, 1.0 - ((review_ratio - 0.4) / 0.6))

        range_span = max_classes - min_classes
        deviation = abs(num_classes - ideal_classes) / (range_span / 2) if range_span != 0 else 0.0
        class_count_score = max(0.0, 1.0 - deviation)

        fitness = (coverage_score * 0.4) + (balance_score * 0.3) + (class_count_score * 0.3)
        logging.debug(f"📊 Evaluated fitness -> Coverage: {coverage_score:.2f}, Balance: {balance_score:.2f}, ClassCount: {class_count_score:.2f}, Final: {fitness:.4f}")
        return fitness

    def evaluate_fitness_with_scores(self, class_distribution, avg_score, constraints):
        """
        Evalúa fitness considerando:
        - validez del plan (repasos coherentes)
        - fitness rápido
        - promedio de scores
        """
    
        fast_fitness = self.evaluate_class_distribution_fast(class_distribution, constraints)

        avg_score = max(0.0, min(avg_score, 1.0))  

        weight_score = 0.5
        final_fitness = (fast_fitness * (1 - weight_score)) + (avg_score * weight_score)

        logging.info(f"🎯 Fitness con score -> Fast: {fast_fitness:.4f}, Score: {avg_score:.4f}, Final: {final_fitness:.4f}")

        return final_fitness


    def tournament_selection(self, population):
        tournament = random.sample(population, self.tournament_size)
        tournament.sort(key=lambda x: x["fitness"], reverse=True)
        winner = tournament[0]
        logging.info(f"🏆 Selección torneo -> Fitness ganador: {winner['fitness']:.4f}")
        return winner["distribution"]

    def crossover(self, parent1, parent2):
        sessions1 = parent1["sessions"]
        sessions2 = parent2["sessions"]
        if len(sessions1) < 2 or len(sessions2) < 2:
            logging.warning("⚠️ Crossover skipped (parents too small)")
            return parent1

        cut_point1 = random.randint(1, len(sessions1)-1)
        cut_point2 = random.randint(1, len(sessions2)-1)
        child_sessions = sessions1[:cut_point1] + sessions2[cut_point2:]

        taught_subtopics = set(s for sess in child_sessions if sess["type"] == "content" for s in sess["subtopics"])
        all_subtopics = taught_subtopics.copy()
        coverage = len(taught_subtopics) / len(all_subtopics) if all_subtopics else 0
        logging.info(f"🔀 Crossover -> Cut points {cut_point1}, {cut_point2}, Coverage: {coverage:.2f}")

        return {"num_classes": len(child_sessions), "coverage": coverage, "sessions": child_sessions}

    def mutate(self, distribution, all_subtopics, mutation_rate=0.1):
        if random.random() > mutation_rate or not distribution["sessions"]:
            return distribution
        content_sessions = [s for s in distribution["sessions"] if s["type"] == "content"]
        if content_sessions:
            session = random.choice(content_sessions)
            if session["subtopics"]:
                idx = random.randint(0, len(session["subtopics"]) - 1)
                old_subtopic = session["subtopics"][idx]
                available_subtopics = [s for s in all_subtopics if s != old_subtopic]
                if available_subtopics:
                    new_subtopic = random.choice(available_subtopics)
                    session["subtopics"][idx] = new_subtopic
                    logging.info(f"🔬 Mutación: {old_subtopic} -> {new_subtopic}")
        taught_subtopics = set(s for sess in distribution["sessions"] if sess["type"] == "content" for s in sess["subtopics"])
        all_possible_subtopics = set(all_subtopics)
        distribution["coverage"] = len(taught_subtopics) / len(all_possible_subtopics) if all_possible_subtopics else 0.0
        return distribution
