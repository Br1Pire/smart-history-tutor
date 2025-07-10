import random
import numpy as np
import datetime
from src import config
from src.core.document_manager import DocumentManager
from src.core.faiss_manager import FaissManager
from src.agents.generator_agent import Generator
from src.agents.vectorizer_agent import Vectorizer
from src.agents.retriever_agent import Retriever
from src.agents.student_agent import StudentAgent

def generate_timestamp():
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    return timestamp_str

def create_student(name, learning_rate_range, skip_probability_range, forgetting_rate = None, detail_preference=None, noise_factor=None, motivation=None, state=None, environment=None):
    """
    Crea un estudiante con todos sus componentes asociados inicializados.
    """

    learning_rate = random.uniform(*learning_rate_range)
    skip_probability = random.uniform(*skip_probability_range)

    if forgetting_rate is None:
        forgetting_rate = random.uniform(0.0, 0.3)

    if detail_preference is None:
        detail_preference = random.choice(["start", "end", "neutral"])

    if noise_factor is None:
        noise_factor = round(random.uniform(0.0, 0.1),3)

    if motivation is None:
        motivation = random.randint(1, 10)
    if state is None:
        state = random.randint(1, 10)
    if environment is None:
        environment = random.randint(1, 10)

    timestamp = generate_timestamp()
    doc_path = config.DOC_FOLDER / f"{name}_{timestamp}.json"
    index_path = config.FAISS_FOLDER / f"{name}_{timestamp}.index"
    id_path = config.FAISS_FOLDER / f"{name}_{timestamp}.json"

    doc_manager = DocumentManager(processed_path=doc_path, prompts_path=config.STUDENTS_PROMPTS_FILE, ids_path=id_path, load=False)
    faiss_manager = FaissManager(index_path=index_path)
    vectorizer = Vectorizer(document_manager=doc_manager, faiss_manager=faiss_manager)
    generator = Generator(document_manager=doc_manager)
    retriever = Retriever(generator=generator, vectorizer=vectorizer, document_manager=doc_manager, text_manager=faiss_manager)

    student = StudentAgent(
        name=name,
        faiss_manager=faiss_manager,
        document_manager=doc_manager,
        retriever=retriever,
        generator=generator,
        vectorizer=vectorizer,
        learning_rate=learning_rate,
        forgetting_rate=forgetting_rate,
        skip_probability=skip_probability,
        detail_preference=detail_preference,
        noise_factor=noise_factor,
        motivation_value=motivation,
        state_value=state,
        environment_value=environment
    )

    return student

def create_multiple_students(amount, students_proportions: tuple = None):
    """
    Crea múltiples estudiantes distribuidos en tres categorías:
    aventajados, normales y con deficiencias.

    Args:
        amount (int): cantidad total de estudiantes.
        students_proportions (tuple): (aventajados, normales, deficiencias) en proporción.

    Returns:
        list of StudentAgent
    """

    if students_proportions is None: balanced_counts = (None, None, None)
    else: balanced_counts = balance_percentages(students_proportions, amount)

    names_pool = ["Carlos", "Ana", "Luis", "María", "Jorge", "Carmen", "Pedro", "Laura", "José", "Paula",
                  "Diego", "Lucía", "Antonio", "Isabel", "Miguel", "Andrea", "Daniel", "Sofía", "Raúl", "Marta",
                  "Rubén", "Elena", "Jesús", "Sara", "Óscar", "Patricia", "Iván", "Cristina", "Sergio", "Clara",
                  "Hugo", "Valeria", "David", "Noelia", "Alberto", "Esther", "Alejandro", "Beatriz", "Francisco",
                  "Eva", "Victor", "Susana", "Javier", "Raquel", "Adrián", "Irene", "Fernando", "Nuria", "Pablo", "Rosa"]

    students = []

    categories = ["aventajados", "normales", "deficientes"]

    for cat, count in zip(categories, balanced_counts):
        for _ in range(count):

            if cat == "aventajados":
                learning_rate_range = (0.8, 1.0)
                skip_probability_range = (0.0, 0.15)
            elif cat == "normales":
                learning_rate_range = (0.6, 0.8)
                skip_probability_range = (0.15, 0.3)
            else:  # deficientes
                learning_rate_range = (0.4, 0.6)
                skip_probability_range = (0.3, 0.5)

            if names_pool:
                name = names_pool.pop(random.randint(0, len(names_pool)-1))
            else:
                name = f"Student_{len(students)+1}"

            student = create_student(
                name=name,
                learning_rate_range=learning_rate_range,
                skip_probability_range=skip_probability_range
            )
            students.append(student)

    suma = sum(x.skip_probability for x in students)

    for student in students:
        total = suma-student.skip_probability
        avg = total * 20/len(students)
        student.update_environment(round(avg,2))
        
        

    return students

def balance_percentages(percentages: tuple, quantity: int) -> tuple:
    if not isinstance(quantity, int) or quantity < 0:
        raise ValueError("La 'quantity' debe ser un número entero no negativo.")

    if len(percentages) != 3:
        raise ValueError("La tupla debe contener exactamente 3 elementos.")

    calculated_percentages_float = list(percentages)
    
    existing_sum = 0
    none_indices = []

    for i, p in enumerate(calculated_percentages_float):
        if p is not None:
            if not (0 <= p <= 100):
                raise ValueError(f"El porcentaje {p} en la posición {i} no está en el rango [0, 100].")
            existing_sum += p
        else:
            none_indices.append(i)

    if existing_sum > 100:
        raise ValueError(f"La suma de los porcentajes numéricos ({existing_sum}%) excede 100%.")

    remaining_percentage = 100 - existing_sum

    if not none_indices:
        if abs(remaining_percentage) > 0.0001:
            raise ValueError(f"No hay 'None' para equilibrar y la suma existente ({existing_sum}) no es 100%.")
    else:
        num_none = len(none_indices)
        if num_none == 1:
            calculated_percentages_float[none_indices[0]] = remaining_percentage
        else:
            weights = [random.random() for _ in range(num_none)]
            sum_weights = sum(weights)
            
            for i, idx in enumerate(none_indices):
                proporcion = weights[i] / sum_weights
                amount_to_add = remaining_percentage * proporcion
                calculated_percentages_float[idx] = amount_to_add
                
    balanced_percentages_int = [int(round(p, 0)) for p in calculated_percentages_float]
    
    current_sum_integers = sum(balanced_percentages_int)
    difference_for_percentages = 100 - current_sum_integers

    if difference_for_percentages != 0:
        if any(p > 0 for p in balanced_percentages_int):
            index_to_adjust = balanced_percentages_int.index(max(balanced_percentages_int))
        else:
            index_to_adjust = 0
        balanced_percentages_int[index_to_adjust] += difference_for_percentages

    distributed_values_float = [ (p / 100) * quantity for p in balanced_percentages_int ]
    
    final_distributed_values = [int(round(val, 0)) for val in distributed_values_float]

    current_distributed_sum = sum(final_distributed_values)
    difference_for_quantity = quantity - current_distributed_sum

    if difference_for_quantity != 0:
        if any(val > 0 for val in final_distributed_values):
            sorted_indices = sorted(range(len(final_distributed_values)), key=lambda k: final_distributed_values[k], reverse=True)
            
            for i in range(abs(difference_for_quantity)):
                if difference_for_quantity > 0:
                    final_distributed_values[sorted_indices[i % 3]] += 1
                else:
                    final_distributed_values[sorted_indices[i % 3]] -= 1
        else:
            final_distributed_values[0] += difference_for_quantity

    return tuple(final_distributed_values)

