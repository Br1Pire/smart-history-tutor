import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

motivation = ctrl.Antecedent(np.arange(0, 11, 1), 'motivation')
state = ctrl.Antecedent(np.arange(0, 11, 1), 'state')
environment = ctrl.Antecedent(np.arange(0, 11, 1), 'environment')
base_lr = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'base_learning_rate')
base_sp = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'base_skip_probability')

learning_rate = ctrl.Consequent(np.arange(0.4, 1.01, 0.01), 'learning_rate')
skip_prob = ctrl.Consequent(np.arange(0, 0.51, 0.01), 'skip_probability')

motivation['low'] = fuzz.trapmf(motivation.universe, [0, 0, 2, 5])
motivation['medium'] = fuzz.trimf(motivation.universe, [2, 5, 8])
motivation['high'] = fuzz.trapmf(motivation.universe, [5, 8, 10, 10])

state['bad'] = fuzz.trapmf(state.universe, [0, 0, 2, 5])
state['normal'] = fuzz.trimf(state.universe, [2, 5, 8])
state['good'] = fuzz.trapmf(state.universe, [5, 8, 10, 10])

environment['focused'] = fuzz.trapmf(environment.universe, [0, 0, 2, 5])
environment['normal'] = fuzz.trimf(environment.universe, [2, 5, 8])
environment['distracted'] = fuzz.trapmf(environment.universe, [5, 8, 10, 10])

base_lr['low'] = fuzz.trapmf(base_lr.universe, [0, 0, 0.25, 0.5])
base_lr['medium'] = fuzz.trimf(base_lr.universe, [0.25, 0.5, 0.75])
base_lr['high'] = fuzz.trapmf(base_lr.universe, [0.5, 0.75, 1, 1])

base_sp['low'] = fuzz.trapmf(base_sp.universe, [0, 0, 0.25, 0.5])
base_sp['medium'] = fuzz.trimf(base_sp.universe, [0.25, 0.5, 0.75])
base_sp['high'] = fuzz.trapmf(base_sp.universe, [0.5, 0.75, 1, 1])

learning_rate['low'] = fuzz.trapmf(learning_rate.universe, [0.4, 0.4, 0.55, 0.7])
learning_rate['medium'] = fuzz.trimf(learning_rate.universe, [0.6, 0.7, 0.8])
learning_rate['high'] = fuzz.trapmf(learning_rate.universe, [0.7, 0.85, 1, 1])

skip_prob['low'] = fuzz.trapmf(skip_prob.universe, [0, 0, 0.15, 0.20])
skip_prob['medium'] = fuzz.trimf(skip_prob.universe, [0.1, 0.2, 0.3])
skip_prob['high'] = fuzz.trapmf(skip_prob.universe, [0.2, 0.3, 0.5, 0.5])

rules = [

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

    ctrl.Rule(base_lr['medium'], learning_rate['medium']),

    ctrl.Rule(base_sp['high'] | environment['distracted'], skip_prob['high']),
    ctrl.Rule(base_sp['medium'] & environment['normal'], skip_prob['medium']),
    ctrl.Rule(base_sp['low'] & environment['focused'], skip_prob['low']),

    ctrl.Rule(base_sp['high'] | state['bad'], skip_prob['high']),
    ctrl.Rule(base_sp['medium'] & state['normal'], skip_prob['medium']),
    ctrl.Rule(base_sp['low'] & state['good'], skip_prob['low']),

    ctrl.Rule(motivation['low'] | environment['distracted'], skip_prob['high']),
    ctrl.Rule(motivation['medium'] & environment['normal'], skip_prob['medium']),
    ctrl.Rule(motivation['high'] & environment['focused'], skip_prob['low']),

    ctrl.Rule(base_sp['medium'], skip_prob['medium'])
]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

def calculate_learning_and_skip(base_lr_value, base_sp_value, motivation_value, state_value, environment_value):
    sim.input['base_learning_rate'] = base_lr_value
    sim.input['base_skip_probability'] = base_sp_value
    sim.input['motivation'] = motivation_value
    sim.input['state'] = state_value
    sim.input['environment'] = environment_value

    sim.compute()

    lr_output = sim.output['learning_rate']
    sp_output = sim.output['skip_probability']

    if np.isnan(lr_output):
        lr_output = base_lr_value
    if np.isnan(sp_output):
        sp_output = base_sp_value

    return lr_output, sp_output


if __name__ == "__main__":
    lr, sp = calculate_learning_and_skip(
        base_lr_value=0.8,
        base_sp_value=0.2,
        motivation_value=0,
        state_value=0,
        environment_value=0
    )
    print(f"Learning Rate: {lr:.2f}, Skip Probability: {sp:.2f}")
