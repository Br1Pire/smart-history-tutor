import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

learning_rate = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'learning_rate')
skip_probability = ctrl.Antecedent(np.arange(0, 0.51, 0.01), 'skip_probability')
score = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'score')

delta_motivation = ctrl.Consequent(np.arange(-5, 5.1, 0.1), 'delta_motivation')

learning_rate['low'] = fuzz.trapmf(learning_rate.universe, [0, 0, 0.2, 0.5])
learning_rate['medium'] = fuzz.trimf(learning_rate.universe, [0.35, 0.6, 0.85])
learning_rate['high'] = fuzz.trapmf(learning_rate.universe, [0.7, 0.8, 1.0, 1.0])

skip_probability['low'] = fuzz.trapmf(skip_probability.universe, [0, 0, 0.1, 0.3])
skip_probability['medium'] = fuzz.trimf(skip_probability.universe, [0.15, 0.3, 0.45])
skip_probability['high'] = fuzz.trapmf(skip_probability.universe, [0.3, 0.4, 0.5, 0.5])

score['low'] = fuzz.trapmf(score.universe, [0, 0, 0.4, 0.5])
score['medium'] = fuzz.trimf(score.universe, [0.4, 0.5, 0.6])
score['high'] = fuzz.trapmf(score.universe, [0.5, 0.7, 1.0, 1.0])

delta_motivation['negative_large'] = fuzz.trapmf(delta_motivation.universe, [-5.0, -5.0, -4.0, -2.5])
delta_motivation['negative_small'] = fuzz.trimf(delta_motivation.universe, [-3.0, -1.5, 0.0])
delta_motivation['zero'] = fuzz.trimf(delta_motivation.universe, [-0.5, 0.0, 0.5])
delta_motivation['positive_small'] = fuzz.trimf(delta_motivation.universe, [0.0, 1.5, 3.0])
delta_motivation['positive_large'] = fuzz.trapmf(delta_motivation.universe, [2.5, 4.0, 5.0, 5.0])

rules = [

    ctrl.Rule(learning_rate['high'] & skip_probability['low'] & score['high'], delta_motivation['positive_small']),
    ctrl.Rule(learning_rate['high'] & skip_probability['low'] & score['medium'], delta_motivation['zero']),
    ctrl.Rule(learning_rate['high'] & skip_probability['low'] & score['low'], delta_motivation['negative_large']),

    ctrl.Rule(learning_rate['low'] & skip_probability['high'] & score['high'], delta_motivation['positive_large']),
    ctrl.Rule(learning_rate['low'] & skip_probability['high'] & score['medium'], delta_motivation['positive_small']),
    ctrl.Rule(learning_rate['low'] & skip_probability['high'] & score['low'], delta_motivation['negative_small']),

    ctrl.Rule(learning_rate['medium'] & skip_probability['medium'] & score['high'], delta_motivation['positive_small']),
    ctrl.Rule(learning_rate['medium'] & skip_probability['medium'] & score['medium'], delta_motivation['zero']),
    ctrl.Rule(learning_rate['medium'] & skip_probability['medium'] & score['low'], delta_motivation['negative_small']),

    ctrl.Rule(learning_rate['high'] & skip_probability['medium'] & score['high'], delta_motivation['positive_small']),
    ctrl.Rule(learning_rate['high'] & skip_probability['medium'] & score['medium'], delta_motivation['zero']),
    ctrl.Rule(learning_rate['high'] & skip_probability['medium'] & score['low'], delta_motivation['negative_large']),

    ctrl.Rule(learning_rate['high'] & skip_probability['high'] & score['high'], delta_motivation['positive_small']),
    ctrl.Rule(learning_rate['high'] & skip_probability['high'] & score['medium'], delta_motivation['zero']),
    ctrl.Rule(learning_rate['high'] & skip_probability['high'] & score['low'], delta_motivation['negative_large']),

    ctrl.Rule(learning_rate['medium'] & skip_probability['low'] & score['high'], delta_motivation['positive_small']),
    ctrl.Rule(learning_rate['medium'] & skip_probability['low'] & score['medium'], delta_motivation['zero']),
    ctrl.Rule(learning_rate['medium'] & skip_probability['low'] & score['low'], delta_motivation['negative_small']),

    ctrl.Rule(learning_rate['medium'] & skip_probability['high'] & score['high'], delta_motivation['positive_small']),
    ctrl.Rule(learning_rate['medium'] & skip_probability['high'] & score['medium'], delta_motivation['zero']),
    ctrl.Rule(learning_rate['medium'] & skip_probability['high'] & score['low'], delta_motivation['negative_small']),

    ctrl.Rule(learning_rate['low'] & skip_probability['low'] & score['high'], delta_motivation['positive_small']),
    ctrl.Rule(learning_rate['low'] & skip_probability['low'] & score['medium'], delta_motivation['zero']),
    ctrl.Rule(learning_rate['low'] & skip_probability['low'] & score['low'], delta_motivation['negative_small']),

    ctrl.Rule(learning_rate['low'] & skip_probability['medium'] & score['high'], delta_motivation['positive_large']),
    ctrl.Rule(learning_rate['low'] & skip_probability['medium'] & score['medium'], delta_motivation['positive_small']),
    ctrl.Rule(learning_rate['low'] & skip_probability['medium'] & score['low'], delta_motivation['negative_small'])        
]


motivation_ctrl = ctrl.ControlSystem(rules)
motivation_simulator = ctrl.ControlSystemSimulation(motivation_ctrl)

def calculate_delta_motivation(lr, sp, sc):
    motivation_simulator.input['learning_rate'] = lr
    motivation_simulator.input['skip_probability'] = sp
    motivation_simulator.input['score'] = sc
    motivation_simulator.compute()
    return motivation_simulator.output['delta_motivation']

if __name__ == "__main__":
    lr = 0.8  
    sp = 0.2
    sc = 0.5  

    delta = calculate_delta_motivation(lr, sp, sc)
    print(f"✅ Cambio en motivación: {delta:.3f}")
