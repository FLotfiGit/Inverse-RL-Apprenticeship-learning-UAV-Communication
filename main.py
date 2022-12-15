"""
Created on January 26th, 2020
@author:    Alireza Shamsoshoara
@Project:   UAV communication using apprenticeship learning via Inverse Reinforcement Learning (IRL)
            Paper: ### TODO
            Arxiv: ### TODO
            Github: https://github.com/AlirezaShamsoshoara/Inverse-RL-Apprenticeship-learning-UAV-Communication
            YouTube Link: ### TODO
@Northern Arizona University
This project is developed and tested with Python 3.6 using pycharm on Ubuntu 18.04 LTS machine


@versions:   xgboost 1.4.2
"""

#################################
# Main File
#################################

# ############# import libraries
# General Modules

# Customized Modules
from utils import UAV
from config import Mode
from qlearning import qrl
from deeprl import deep_rl
from location import plotues
from utils import create_ues
from utils import create_cells
from expert import expert_policy
from location import plothexagon
from config import Config_General
from utils import find_closest_cell
from shortestpath import short_path
from randompolicy import random_action
from evaluation import evaluation_error
from inverserlSGD import inverse_rl_sgd
from inverserlDQN import inverse_rl_dqn
from behavioral import behavioral_cloning
from evaluation import evaluation_training
from evaluation import evaluation_scenario
from evaluation import inverse_rl_hyper_distance

#########################################################
# General Parameters
Altitude = Config_General.get('Altitude')

#########################################################
# Main


def main():
    print(" ..... Running:")
    print("UAV communication using apprenticeship learning via Inverse Reinforcement Learning (IRL)")
    uav = UAV(x_loc=0, y_loc=0, z_loc=Altitude, cell_id=0)
    v_coord_cells, h_coord_cells, cell_ids, fig_cells, ax_cells, coordinates = plothexagon()
    fig_ues, ax_ues, x_coord_ues, y_coord_ues = plotues(fig_cells, ax_cells, cell_ids, h_coord_cells, v_coord_cells)
    ue_cell_ids = find_closest_cell(h_coord_cells, v_coord_cells, x_coord_ues, y_coord_ues)
    ues_objects = create_ues(x_coord_ues, y_coord_ues, ue_cell_ids)
    cells_objects = create_cells(h_coord_cells, v_coord_cells, cell_ids, ue_cell_ids, coordinates)

    return uav, ues_objects, ax_ues, cells_objects


if __name__ == "__main__":
    uav_main, ues_objects_main, ax_ues_main, cells_objects_main = main()

    if Mode == "Expert":
        expert_policy(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "IRL_SGD":
        inverse_rl_sgd(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "IRL_DQN":
        inverse_rl_dqn(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "DRL":
        deep_rl()
    elif Mode == "QRL":
        qrl()
    elif Mode == "BC":
        behavioral_cloning(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "Shortest":
        short_path(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "Random":
        random_action(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "ResultsIRL":
        inverse_rl_hyper_distance()
    elif Mode == "EvaluationTraining":
        evaluation_training(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "EvaluationScenario":
        evaluation_scenario(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "EvaluationError":
        evaluation_error(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    else:
        print("Mode is not correct")
        exit(' ........... Exit: wrong chosen mode')
