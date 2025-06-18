import random
import numpy as np
import logging
import os
from src.config import LOG_FILES, MAX_ITER

# Configuración de logs
LOG_FILE = LOG_FILES["chunking"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)




def chunk_section_text_metaheuristic(section_name, sentences, max_chunk_size, min_chunk_size):
    """
    Aplica un algoritmo metaheurístico para dividir una sección en chunks óptimos.

    Args:
        section_name (str or None): Nombre de la sección.
        sentences (list): Lista de oraciones de la sección.
        max_chunk_size (int): Tamaño máximo del chunk (en tokens).
        min_chunk_size (int): Tamaño mínimo del chunk (en tokens).

    Returns:
        tuple: Lista de tuplas (sección, chunk) y mejor costo encontrado.
    """
    sent_lengths = [len(sent.split()) for sent in sentences]
    logging.info(f"🚀 Iniciando chunking metaheurístico para sección: '{section_name or 'General'}' con {len(sentences)} oraciones.")

    def initial_solution():
        """Genera una solución inicial dividiendo el texto por longitud máxima."""
        cuts = []
        current_len = 0
        for i, slen in enumerate(sent_lengths):
            current_len += slen
            if current_len > max_chunk_size:
                cuts.append(i)
                current_len = slen
        logging.info(f"✅ Solución inicial generada con {len(cuts)} cortes.")
        return cuts

    def cost(cuts):
        """
        Calcula el costo de una solución dada en función de tamaño y varianza.

        Args:
            cuts (list): Lista de posiciones de corte.

        Returns:
            float: Costo de la solución.
        """
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
        total_cost = sum(penalties) + var_penalty
        return total_cost

    def neighbor(cuts):
        """
        Genera una solución vecina moviendo un corte al azar.

        Args:
            cuts (list): Lista actual de cortes.

        Returns:
            list: Nueva lista de cortes.
        """
        if not cuts:
            return cuts
        new_cuts = cuts[:]
        idx = random.randint(0, len(new_cuts)-1)
        direction = random.choice([-1, 1])
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

    for iteration in range(MAX_ITER):
        new = neighbor(current)
        new_cost = cost(new)
        if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
            current = new
            current_cost = new_cost
            if new_cost < best_cost:
                best = new
                best_cost = new_cost
        temperature = max(temperature * (1 - cooling_rate), 1e-6)
        if iteration % 1000 == 0:
            logging.info(f"🔄 Iteración {iteration}: mejor costo actual = {best_cost:.4f}")

    indices = [0] + best + [len(sentences)]
    chunk_texts = []
    for i in range(len(indices)-1):
        chunk_sents = sentences[indices[i]:indices[i+1]]
        chunk_text = ' '.join(chunk_sents)
        chunk_texts.append((section_name, chunk_text))

    logging.info(f"🏁 Chunking completado: {len(chunk_texts)} chunks generados. Mejor costo: {best_cost:.4f}")
    return chunk_texts, best_cost
