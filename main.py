import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

#Criando as variáveis
qualidade_da_comida = ctrl.Antecedent(np.arange(0, 11, 1), 'qualidade_da_comida')
atendimento = ctrl.Antecedent(np.arange(0, 11, 1), 'atendimento')
gorjeta = ctrl.Consequent(np.arange(0, 31, 1), 'gorjeta')

qualidade_da_comida.automf(names=['Ruim', 'Excelente'])


#Criando as funções de pertinência
atendimento['Ruim'] = fuzz.gaussmf(atendimento.universe, 0, 1.25)
atendimento['Bom'] = fuzz.gaussmf(atendimento.universe, 5, 1.25)
atendimento['Excelente'] = fuzz.smf(atendimento.universe, 5, 10)

qualidade_da_comida['Ruim'] = fuzz.trapmf(qualidade_da_comida.universe, [0, 0, 1.25, 3.75])
qualidade_da_comida['Excelente'] = fuzz.trapmf(qualidade_da_comida.universe, [6.25, 8.75, 10, 10])

gorjeta['Baixa'] = fuzz.trimf(gorjeta.universe, [0, 5, 10])
gorjeta['Média'] = fuzz.trimf(gorjeta.universe, [10, 15, 20])
gorjeta['Alta'] = fuzz.trimf(gorjeta.universe, [20, 25, 30])


#Criando as regras e o controle
rule1 = ctrl.Rule(atendimento['Excelente'] | qualidade_da_comida['Excelente'], gorjeta['Alta'])
rule2 = ctrl.Rule(atendimento['Bom'], gorjeta['Média'])
rule3 = ctrl.Rule(atendimento['Ruim'] & qualidade_da_comida['Ruim'], gorjeta['Baixa'])

gorjeta_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
gorjeta_simulador = ctrl.ControlSystemSimulation(gorjeta_ctrl)


#Selecionando as entradas
gorjeta_simulador.input['qualidade_da_comida'] = 3.5
gorjeta_simulador.input['atendimento'] = 1.0

#Mostrando a saída e os gráficos
gorjeta_simulador.compute()
print("Gorjeta = R$ {}".format(gorjeta_simulador.output['gorjeta']))

qualidade_da_comida.view(sim=gorjeta_simulador)
atendimento.view(sim=gorjeta_simulador)
gorjeta.view(sim=gorjeta_simulador)

