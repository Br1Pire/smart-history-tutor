import random
import numpy as np

CHUNK_SIZE = 100  # tamaño máximo en tokens (o palabras)
MIN_CHUNK_LENGTH = 30  # tamaño mínimo en tokens (o palabras)
MAX_ITER = 5000  # número de iteraciones de la metaheurística

def chunk_section_text_metaheuristic(section_name, sentences, max_chunk_size=CHUNK_SIZE, min_chunk_size=MIN_CHUNK_LENGTH):
    sent_lengths = [len(sent.split()) for sent in sentences]

    # Crea una solución inicial: cortes secuenciales
    def initial_solution():
        cuts = []
        current_len = 0
        for i, slen in enumerate(sent_lengths):
            current_len += slen
            if current_len > max_chunk_size:
                cuts.append(i)
                current_len = slen
        return cuts

    # Calcula el costo de los cortes
    def cost(cuts):
        indices = [0] + cuts + [len(sentences)]
        penalties = []
        sizes = []
        for i in range(len(indices)-1):
            size = sum(sent_lengths[indices[i]:indices[i+1]])
            sizes.append(size)
            if size < min_chunk_size:
                penalties.append((min_chunk_size - size) * 2)
            if size > max_chunk_size:
                penalties.append((size - max_chunk_size) * 2)
        var_penalty = np.var(sizes) * 0.5 if len(sizes) > 1 else 0
        return sum(penalties) + var_penalty

    # Crea un vecino: mueve un punto de corte un poco
    def neighbor(cuts):
        if not cuts:
            return cuts
        new_cuts = cuts[:]
        idx = random.randint(0, len(new_cuts)-1)
        direction = random.choice([-1, 1])
        # Asegúrate de no salirse del rango y no cruzar otro corte
        new_pos = new_cuts[idx] + direction
        if new_pos <= 0 or new_pos >= len(sentences):
            return cuts
        if idx > 0 and new_pos <= new_cuts[idx-1]:
            return cuts
        if idx < len(new_cuts)-1 and new_pos >= new_cuts[idx+1]:
            return cuts
        new_cuts[idx] = new_pos
        return new_cuts

    current = initial_solution()
    best = current
    current_cost = cost(current)
    best_cost = current_cost

    temperature = 1.0
    cooling_rate = 0.0002

    for _ in range(MAX_ITER):
        new = neighbor(current)
        new_cost = cost(new)
        if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
            current = new
            current_cost = new_cost
            if new_cost < best_cost:
                best = new
                best_cost = new_cost
        temperature = max(temperature * (1 - cooling_rate), 1e-6)

    # Reconstruir chunks
    indices = [0] + best + [len(sentences)]
    chunk_texts = []
    for i in range(len(indices)-1):
        chunk_sents = sentences[indices[i]:indices[i+1]]
        chunk_text = ' '.join(chunk_sents)
        chunk_texts.append((section_name, chunk_text))

    return chunk_texts, best_cost
