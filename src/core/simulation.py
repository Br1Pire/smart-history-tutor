import logging
import numpy as np
from src.agents.student_agent import StudentAgent
from src.agents.tutor_agent import Tutor

def run_simulation_cycle(plan, tutor: Tutor, students: list[StudentAgent]):
    """
    Ejecuta un ciclo de simulación con:
    - Enseñanza a estudiantes (actualiza su conocimiento si se modela).
    - Aplicación de tests al finalizar.
    - Evaluación de los resultados.

    Args:
        plan (dict): plan de clases completo (con clases y tests ya generados).
        student_agent (StudentAgent): agente del estudiante.

    Returns:
        dict: resultados por estudiante, score promedio global, detalles.
    """

    for session in plan['sessions']:
        for student in students:
            student.take_session(session)  
            student.forget() 

    for student in students:
        student.persist_faiss()
    
    student_tests = []

    for student in students:
        answer = student.answer_test(plan['test'])
        student_tests.append(answer)


    evaluation_results = tutor.evaluate_student_tests(student_tests, plan['test'])

    average_scores = [student["average_score"] for student in evaluation_results]
    global_avg = np.mean(average_scores)

    for student, test in zip(students, evaluation_results):
        student.update_motivation(test['average_score'])
        student.clear_memory()

    

    logging.info(f"✅ Simulación completada. Global average score: {global_avg:.4f}")

    return {
        "evaluation_results": evaluation_results,
        "global_average_score": float(global_avg)
    }
