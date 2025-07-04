import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ===============================
# I. Definición de universos
# ===============================

# Inputs
motivation = ctrl.Antecedent(np.arange(0, 11, 1), 'motivation')
state = ctrl.Antecedent(np.arange(0, 11, 1), 'state')
environment = ctrl.Antecedent(np.arange(0, 11, 1), 'environment')
base_lr = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'base_learning_rate')
base_sp = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'base_skip_probability')

# Outputs
learning_rate = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'learning_rate')
skip_prob = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'skip_probability')

# ===============================
# II. Funciones de membresía
# ===============================

# Motivation
motivation['low'] = fuzz.trimf(motivation.universe, [0, 0, 5])
motivation['medium'] = fuzz.trimf(motivation.universe, [2, 5, 8])
motivation['high'] = fuzz.trimf(motivation.universe, [5, 10, 10])

# State
state['bad'] = fuzz.trimf(state.universe, [0, 0, 5])
state['normal'] = fuzz.trimf(state.universe, [2, 5, 8])
state['good'] = fuzz.trimf(state.universe, [5, 10, 10])

# Environment
environment['focused'] = fuzz.trimf(environment.universe, [0, 0, 5])
environment['normal'] = fuzz.trimf(environment.universe, [2, 5, 8])
environment['distracted'] = fuzz.trimf(environment.universe, [5, 10, 10])

# Base Learning Rate
base_lr['low'] = fuzz.trimf(base_lr.universe, [0, 0, 0.5])
base_lr['medium'] = fuzz.trimf(base_lr.universe, [0.25, 0.5, 0.75])
base_lr['high'] = fuzz.trimf(base_lr.universe, [0.5, 1, 1])

# Base Skip Probability
base_sp['low'] = fuzz.trimf(base_sp.universe, [0, 0, 0.5])
base_sp['medium'] = fuzz.trimf(base_sp.universe, [0.25, 0.5, 0.75])
base_sp['high'] = fuzz.trimf(base_sp.universe, [0.5, 1, 1])

# Learning Rate output
learning_rate['low'] = fuzz.trimf(learning_rate.universe, [0, 0, 0.5])
learning_rate['medium'] = fuzz.trimf(learning_rate.universe, [0.25, 0.5, 0.75])
learning_rate['high'] = fuzz.trimf(learning_rate.universe, [0.5, 1, 1])

# Skip Probability output
skip_prob['low'] = fuzz.trimf(skip_prob.universe, [0, 0, 0.5])
skip_prob['medium'] = fuzz.trimf(skip_prob.universe, [0.25, 0.5, 0.75])
skip_prob['high'] = fuzz.trimf(skip_prob.universe, [0.5, 1, 1])

# ===============================
# III. Reglas de inferencia
# ===============================

rules = [

    # --- Learning Rate ---
    ctrl.Rule(base_lr['high'] & motivation['high'], learning_rate['high']),
    ctrl.Rule(base_lr['high'] & motivation['low'], learning_rate['medium']),
    ctrl.Rule(base_lr['low'] | motivation['low'], learning_rate['low']),
    ctrl.Rule(base_lr['medium'] & motivation['medium'], learning_rate['medium']),

    ctrl.Rule(base_lr['high'] & state['good'], learning_rate['high']),
    ctrl.Rule(base_lr['low'] | state['bad'], learning_rate['low']),
    ctrl.Rule(base_lr['medium'] & state['normal'], learning_rate['medium']),

    ctrl.Rule(motivation['high'] | state['good'], learning_rate['high']),
    ctrl.Rule(motivation['low'] | state['bad'], learning_rate['low']),
    ctrl.Rule(motivation['medium'] & state['normal'], learning_rate['medium']),

    ctrl.Rule(base_lr['high'] & environment['focused'], learning_rate['high']),
    ctrl.Rule(base_lr['low'] | environment['distracted'], learning_rate['low']),
    ctrl.Rule(base_lr['medium'] & environment['normal'], learning_rate['medium']),

    # Catch-all
    ctrl.Rule(base_lr['medium'], learning_rate['medium']),

    # --- Skip Probability ---
    ctrl.Rule(base_sp['high'] | environment['distracted'], skip_prob['high']),
    ctrl.Rule(base_sp['medium'] & environment['normal'], skip_prob['medium']),
    ctrl.Rule(base_sp['low'] & environment['focused'], skip_prob['low']),

    ctrl.Rule(base_sp['high'] | state['bad'], skip_prob['high']),
    ctrl.Rule(base_sp['medium'] & state['normal'], skip_prob['medium']),
    ctrl.Rule(base_sp['low'] & state['good'], skip_prob['low']),

    ctrl.Rule(motivation['low'] | environment['distracted'], skip_prob['high']),
    ctrl.Rule(motivation['medium'] & environment['normal'], skip_prob['medium']),
    ctrl.Rule(motivation['high'] & environment['focused'], skip_prob['low']),

    # Catch-all
    ctrl.Rule(base_sp['medium'], skip_prob['medium'])
]

# ===============================
# IV. Sistema de control
# ===============================

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

# ===============================
# V. Función de cálculo
# ===============================

def calculate_learning_and_skip(base_lr_value, base_sp_value, motivation_value, state_value, environment_value):
    sim.input['base_learning_rate'] = base_lr_value
    sim.input['base_skip_probability'] = base_sp_value
    sim.input['motivation'] = motivation_value
    sim.input['state'] = state_value
    sim.input['environment'] = environment_value

    sim.compute()

    # Fallback en caso de NaN
    lr_output = sim.output['learning_rate']
    sp_output = sim.output['skip_probability']

    if np.isnan(lr_output):
        lr_output = base_lr_value
    if np.isnan(sp_output):
        sp_output = base_sp_value

    return lr_output, sp_output

# ===============================
# VI. Ejemplo de uso
# ===============================

if __name__ == "__main__":
    lr, sp = calculate_learning_and_skip(
        base_lr_value=0.7,
        base_sp_value=0.2,
        motivation_value=8,
        state_value=9,
        environment_value=2
    )
    print(f"Learning Rate: {lr:.2f}, Skip Probability: {sp:.2f}")
