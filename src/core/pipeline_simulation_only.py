import json
from src.core.tutor_builder import create_tutor_instance
from src.core.student_builder import create_multiple_students
from src.core.simulation import run_simulation_cycle
from src import config


def run_pipeline(topic, students_amount, students_proportion, subptopic_range, constraints):
    tutor = create_tutor_instance()

    students = create_multiple_students(students_amount,students_proportion)

    subtopics =  tutor.generate_subtopics(topic,subptopic_range)

    subtopics_with_chunks = tutor.retrive_chunks_from_subtopics(subtopics)

    subtopics_with_text = tutor.generate_texts_from_subtopics(subtopics_with_chunks)

    tests = tutor.generate_tests_from_subtopics(subtopics_with_text)

    path = config.DATA_DIR / 'simulation'  

    with open(path / "texts.json", "w", encoding="utf-8") as f:
        json.dump(subtopics_with_text, f, ensure_ascii=False, indent=2)

    with open(path / "tests.json", "w", encoding="utf-8") as f:
        json.dump(tests, f, ensure_ascii=False, indent=2)

    class_distribution = tutor.generate_class_distribution(subtopics, constraints)

    plan = tutor.generate_plan(class_distribution, subtopics_with_text, tests)

    with open(path / 'plan.json', "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    results = run_simulation_cycle(plan, tutor, students)

    path = config.DATA_DIR / 'simulation'  
    with open(path / 'results.json', "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

if __name__ == '__main__':

    topic="Historia Contemporanea"
    students_amount = 3
    students_proportion = (100, 0, 0)
    subptopic_range = (5,5)
    constraints = {
        'num_classes_range': (3,6),
        'coverage': 0.7
    }

    results = run_pipeline(topic, students_amount, students_proportion, subptopic_range, constraints)
    
    
    