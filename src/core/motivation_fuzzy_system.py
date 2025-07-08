import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ====== Definir variables difusas ======
learning_rate = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'learning_rate')
skip_probability = ctrl.Antecedent(np.arange(0, 0.51, 0.01), 'skip_probability')
score = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'score')

delta_motivation = ctrl.Consequent(np.arange(-5, 5.1, 0.1), 'delta_motivation')

# ====== Funciones de membresía ======
learning_rate['low'] = fuzz.trapmf(learning_rate.universe, [0, 0, 0.2, 0.5])
learning_rate['medium'] = fuzz.trimf(learning_rate.universe, [0.35, 0.6, 0.85])
learning_rate['high'] = fuzz.trapmf(learning_rate.universe, [0.7, 0.8, 1.0, 1.0])

skip_probability['low'] = fuzz.trapmf(skip_probability.universe, [0, 0, 0.1, 0.3])
skip_probability['medium'] = fuzz.trimf(skip_probability.universe, [0.15, 0.3, 0.45])
skip_probability['high'] = fuzz.trapmf(skip_probability.universe, [0.3, 0.4, 0.5, 0.5])

score['low'] = fuzz.trapmf(score.universe, [0, 0, 0.3, 0.6])
score['medium'] = fuzz.trimf(score.universe, [0.5, 0.7, 0.8])
score['high'] = fuzz.trapmf(score.universe, [0.75, 0.9, 1.0, 1.0])

delta_motivation['negative_large'] = fuzz.trapmf(delta_motivation.universe, [-5.0, -5.0, -4.0, -2.5])
delta_motivation['negative_small'] = fuzz.trimf(delta_motivation.universe, [-3.0, -1.5, 0.0])
delta_motivation['zero'] = fuzz.trimf(delta_motivation.universe, [-0.5, 0.0, 0.5])
delta_motivation['positive_small'] = fuzz.trimf(delta_motivation.universe, [0.0, 1.5, 3.0])
delta_motivation['positive_large'] = fuzz.trapmf(delta_motivation.universe, [2.5, 4.0, 5.0, 5.0])

# ====== Reglas difusas ======

rule1 = ctrl.Rule(learning_rate['high'] & skip_probability['low'] & score['high'], delta_motivation['positive_small'])
rule2 = ctrl.Rule(learning_rate['high'] & skip_probability['low'] & score['medium'], delta_motivation['zero'])
rule3 = ctrl.Rule(learning_rate['high'] & skip_probability['low'] & score['low'], delta_motivation['negative_large'])

rule4 = ctrl.Rule(learning_rate['low'] & skip_probability['high'] & score['high'], delta_motivation['positive_large'])
rule5 = ctrl.Rule(learning_rate['low'] & skip_probability['high'] & score['medium'], delta_motivation['positive_small'])
rule6 = ctrl.Rule(learning_rate['low'] & skip_probability['high'] & score['low'], delta_motivation['negative_small'])

rule7 = ctrl.Rule(learning_rate['medium'] & skip_probability['medium'] & score['high'], delta_motivation['positive_small'])
rule8 = ctrl.Rule(learning_rate['medium'] & skip_probability['medium'] & score['medium'], delta_motivation['zero'])
rule9 = ctrl.Rule(learning_rate['medium'] & skip_probability['medium'] & score['low'], delta_motivation['negative_small'])

rule10 = ctrl.Rule(learning_rate['high'] & skip_probability['medium'] & score['high'], delta_motivation['positive_small'])
rule11 = ctrl.Rule(learning_rate['high'] & skip_probability['medium'] & score['medium'], delta_motivation['zero'])
rule12 = ctrl.Rule(learning_rate['high'] & skip_probability['medium'] & score['low'], delta_motivation['negative_large'])

rule13 = ctrl.Rule(learning_rate['high'] & skip_probability['high'] & score['high'], delta_motivation['positive_small'])
rule14 = ctrl.Rule(learning_rate['high'] & skip_probability['high'] & score['medium'], delta_motivation['zero'])
rule15 = ctrl.Rule(learning_rate['high'] & skip_probability['high'] & score['low'], delta_motivation['negative_large'])

rule16 = ctrl.Rule(learning_rate['medium'] & skip_probability['low'] & score['high'], delta_motivation['positive_small'])
rule17 = ctrl.Rule(learning_rate['medium'] & skip_probability['low'] & score['medium'], delta_motivation['zero'])
rule18 = ctrl.Rule(learning_rate['medium'] & skip_probability['low'] & score['low'], delta_motivation['negative_small'])

rule19 = ctrl.Rule(learning_rate['medium'] & skip_probability['high'] & score['high'], delta_motivation['positive_small'])
rule20 = ctrl.Rule(learning_rate['medium'] & skip_probability['high'] & score['medium'], delta_motivation['zero'])
rule21 = ctrl.Rule(learning_rate['medium'] & skip_probability['high'] & score['low'], delta_motivation['negative_small'])

rule22 = ctrl.Rule(learning_rate['low'] & skip_probability['low'] & score['high'], delta_motivation['positive_small'])
rule23 = ctrl.Rule(learning_rate['low'] & skip_probability['low'] & score['medium'], delta_motivation['zero'])
rule24 = ctrl.Rule(learning_rate['low'] & skip_probability['low'] & score['low'], delta_motivation['negative_small'])

rule25 = ctrl.Rule(learning_rate['low'] & skip_probability['medium'] & score['high'], delta_motivation['positive_large'])
rule26 = ctrl.Rule(learning_rate['low'] & skip_probability['medium'] & score['medium'], delta_motivation['positive_small'])
rule27 = ctrl.Rule(learning_rate['low'] & skip_probability['medium'] & score['low'], delta_motivation['negative_small'])



# ====== Sistema de control ======
motivation_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, 
    rule7, rule8, rule9, rule10, rule11, rule12, 
    rule13, rule14, rule15, rule16, rule17, rule18, 
    rule19, rule20, rule21, rule22, rule23, rule24,
    rule25, rule26, rule27
])
motivation_simulator = ctrl.ControlSystemSimulation(motivation_ctrl)

# ====== Función para calcular delta motivación ======
def calculate_delta_motivation(lr, sp, sc):
    motivation_simulator.input['learning_rate'] = lr
    motivation_simulator.input['skip_probability'] = sp
    motivation_simulator.input['score'] = sc
    motivation_simulator.compute()
    return motivation_simulator.output['delta_motivation']

# ====== Ejemplo de uso ======
if __name__ == "__main__":
    lr = 0.8  # alumno aventajado
    sp = 0.2
    sc = 0.5  # nota baja

    delta = calculate_delta_motivation(lr, sp, sc)
    print(f"✅ Cambio en motivación: {delta:.3f}")
