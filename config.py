"""
#################################
# Configuration File
#################################
"""

#########################################################
# import libraries
import numpy as np

#########################################################
# Configuration
Mode = "IRL_SGD" #'Expert'#"IRL_SGD"#"IRL_DQN" # "EvaluationTraining" #"EvaluationScenario"  # #  @fl
# Different Modes {"Expert", "IRL_SGD", "IRL_DQN", "DRL", "QRL", "BC", "Shortest", "Random", "ResultsIRL",
# "EvaluationTraining", "EvaluationScenario", "EvaluationError"}

Config_Flags = {'SAVE_path': True, 'Display_map': False, 'SingleArrow': False, 'SAVE_IRL_DATA': True,
                'SAVE_EXPERT_DATA': True, 'SAVE_IRL_WEIGHT': True, 'SAVE_MODEL_IRL_SGD': True, 'PLOT_RESULTS': True,
                'SAVE_PLOT_PDF': True, 'SAVE_PLOT_FIG': True, 'PRINT_INFO': True, 'LOAD_IRL': False,
                'SAVE_DATA_BC_EXPERT': True, 'SAVE_MODEL_BC': True, 'SAVE_IRL_DATA_DQN': True,
                'SAVE_MODEL_IRL_DQN': True, 'DISABLE_GPU': False}  ## fl ---- GPU enable!
# Config_Flags = {'SAVE_path': True, 'Display_map': True, 'SingleArrow': False, 'SAVE_IRL_DATA': False,
#                 'SAVE_EXPERT_DATA': False, 'SAVE_IRL_WEIGHT': False, 'SAVE_MODEL_IRL_SGD': False, 'PLOT_RESULTS': False,
#                 'SAVE_PLOT_PDF': False, 'SAVE_PLOT_FIG': False, 'PRINT_INFO': False, 'LOAD_IRL': False,
#                 'SAVE_DATA_BC_EXPERT': False, 'SAVE_MODEL_BC': False, 'SAVE_IRL_DATA_DQN': False,
#                 'SAVE_MODEL_IRL_DQN': False, 'DISABLE_GPU': True}

# Possible number of UEs Cluster: 75
# Possible number of Cells: 25
Config_General = {'NUM_UAV': 1, 'Size': 5, 'NUM_CELLS': 25, 'NUM_UEs': 75, 'Radius': 10, 'Loc_delta': 2,
                  'FLOAT_ACCURACY': 6, 'Altitude': 50.0}
Config_requirement = {'dist_limit': Config_General.get('Size') + 3, 'MAX_DISTANCE': 6, 'MIN_UE_NEIGHBORS': 4,
                      'MAX_UE_NEIGHBORS': 29, 'MIN_INTERFERENCE': 0.5123281666343314,
                      'MAX_INTERFERENCE': 14.621335028196711}

Number_of_neighbor_UEs = {'Min': 0, 'Max': 0}

config_movement_step = {'x_step': (Config_General.get('Radius')) * (3./2.),
                        'y_step': (Config_General.get('Radius')) * np.sqrt(3)}

movement_actions_list = [1, 2, 3, 4, 5, 6]  # 1: North, 2: North East, 3: South East, 4: South, 5: South West,
# 6: North West
Config_interference = {'AntennaGain': 100, 'Bandwidth': 50}
Config_Power = {'UE_Tr_power': 2.0, 'UAV_Tr_power': [50.0, 60.0, 80.0, 100.0, 150.0, 200.0], 'UAV_init_energy': 400.0,
                'UAV_mobility_consumption': 10.0}  # Tr power: mW, Energy, Jule
# [50.0, 60.0, 80.0, 100.0, 150.0, 200.0]
# [50.0, 80.0, 100.0, 150.0]

Config_IRL = {'NUM_FEATURES': 5, 'NUM_EPOCHS': 10000, 'NUM_PLAY': 1, 'NUM_TRAJECTORIES_EXPERT': 1000,
              'TRAJECTORY_LENGTH': Config_requirement.get('dist_limit'), 'GAMMA_FEATURES': 0.999,
              'EPSILON_OPTIMIZATION': 0.1, 'EPSILON_GREEDY': 0.1,
              'GAMMA_DISCOUNT': 0.9}  ### @fl   ------->    'EPSILON_OPTIMIZATION': 0.01, 'EPSILON_GREEDY': 0.1, @fl 'NUM_TRAJECTORIES_EXPERT': 1

Config_IRL_DQN = {'NUM_EPOCHS': 10000, 'BUFFER_LENGTH': 10000, 'BATCH_SIZE': 24, 'LEARNING_RATE': 1e-3}

Config_BehavioralCloning = {'NUM_TRAJECTORIES_EXPERT': 10000}

Config_Evaluation = {'NUM_TRAINING': 101}

Config_QRL = {}
Config_DRL = {}

pathDist = 'ConfigData/Cells_%d_Size_%d_UEs_%d' % (Config_General.get('NUM_CELLS'), Config_General.get('Size'),
                                                   Config_General.get('NUM_UEs'))

ExpertPath = "Data/ExpertDemo/"
WeightPath = "Data/Weights/"
WeightPath_DQN = "Data/Weights_DQN/"
InverseRLPath = "Data/InverseRL/"
InverseRLPathDQN = "Data/InverseRL/DQNData/"
ResultPathPDF = "Results/PDF/"
ResultPathFIG = "Results/FIG/"
SGDModelPath = "Data/InverseRL/SGDModel/"
DQNModelPath = "Data/InverseRL/DQNModel/"
ExpertPath_BC = "Data/BehavioralCloning/"
BCModelPath = "Data/BehavioralCloning/Model/"
Config_Path = {'PathDist': pathDist, 'ExpertPath': ExpertPath, 'WeightPath': WeightPath, 'InverseRLPath': InverseRLPath,
               'ResultPathPDF': ResultPathPDF, 'ResultPathFIG': ResultPathFIG, 'SGDModelPath': SGDModelPath,
               'ExpertPath_BC': ExpertPath_BC, 'BCModelPath': BCModelPath, 'WeightPath_DQN': WeightPath_DQN,
               'DQNModelPath': DQNModelPath, 'InverseRLPathDQN': InverseRLPathDQN}
