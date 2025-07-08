def evaluate_plan_fast(plan, constraints):
    """
    Evalúa un plan de clases de forma rápida.

    Args:
        plan (dict): Plan de clases.
        constraints (dict): Incluye num_classes_range y coverage_interval para ajustes dinámicos.

    Returns:
        float: score de fitness.
    """

    coverage = plan["coverage_used"]
    num_classes = plan["num_classes_used"]
    num_reviews = sum(1 for s in plan["sessions"] if s["type"] == "review")

    # 🔢 Parámetros base
    min_classes, max_classes = constraints["num_classes_range"]
    avg_classes = (min_classes + max_classes) / 2

    # 🎯 1. Cobertura: peso moderado
    coverage_score = coverage * 1.0

    # 🎯 2. Equilibrio de clases: penalización por alejarse del promedio
    class_balance_penalty = abs(num_classes - avg_classes) * 0.1

    # 🎯 3. Repasos: penaliza exceso o ausencia, premia proporción moderada
    if num_reviews == 0:
        review_score = -0.3  # penaliza no tener repasos
    else:
        review_ratio = num_reviews / num_classes
        if 0.2 <= review_ratio <= 0.4:
            review_score = +0.3  # premia balance adecuado
        elif review_ratio > 0.5:
            review_score = -0.2  # penaliza exceso
        else:
            review_score = -0.1  # penaliza muy pocos si no llegan a 20%

    # 🎯 4. Penalización por cobertura muy baja (< min_coverage del constraint)
    min_coverage = constraints["coverage"]
    if coverage < min_coverage:
        low_coverage_penalty = -0.5
    else:
        low_coverage_penalty = 0.0

    # ✅ Fitness final
    fitness = coverage_score - class_balance_penalty + review_score + low_coverage_penalty

    return fitness