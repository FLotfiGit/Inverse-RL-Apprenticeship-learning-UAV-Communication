"""
#################################
# Expert Operation functions
#################################
"""

#########################################################
# import libraries
import numpy as np
from copy import deepcopy
from random import randint
from config import Config_IRL
from config import Config_Path
from config import Config_Flags
from config import Config_Power
import matplotlib.pyplot as plt
from location import reset_axes
from location import update_axes
from config import Config_General
from config import Config_requirement
from config import movement_actions_list
from utils import multi_actions_to_action
from config import Number_of_neighbor_UEs

#########################################################
# General Parameters
gamma = Config_IRL.get('GAMMA_FEATURES')
ExpertPath = Config_Path.get('ExpertPath')
num_cells = Config_General.get('NUM_CELLS')
tx_powers = Config_Power.get('UAV_Tr_power')
num_features = Config_IRL.get('NUM_FEATURES')
dist_limit = Config_requirement.get('dist_limit')
MAX_DISTANCE = Config_requirement.get('MAX_DISTANCE')
trajectory_length = Config_IRL.get('TRAJECTORY_LENGTH')
num_trajectories = Config_IRL.get('NUM_TRAJECTORIES_EXPERT')
MIN_UE_NEIGHBORS = Config_requirement.get('MIN_UE_NEIGHBORS')
MAX_UE_NEIGHBORS = Config_requirement.get('MAX_UE_NEIGHBORS')
MIN_INTERFERENCE = Config_requirement.get('MIN_INTERFERENCE')
MAX_INTERFERENCE = Config_requirement.get('MAX_INTERFERENCE')

cell_source = 0
cell_destination = num_cells - 1
#########################################################
# Function definition


def expert_policy(uav, ues_objects, ax_objects, cell_objects):
    episode = 0
    prev_cell = 1
    trajectories = []
    arrow_patch_list = []
    pp = Config_Power['UAV_Tr_power'] #@fl
    cell_to_move = [1,1,2,2,2,2] #@fl
    while episode < num_trajectories:
        trajectory = []
        distance = 0
        done = False
        arrow_patch_list = reset_axes(ax_objects=ax_objects, cell_source=cell_source, cell_destination=cell_destination,
                                      arrow_patch_list=arrow_patch_list)
        uav.uav_reset(cell_objects)
        expert_feature_expectation = np.zeros(num_features, dtype=float)
        ii =0
        while distance < dist_limit and not done:
            cell = uav.get_cell_id()
            current_state = uav.get_cell_id()
            expert_action_mov = cell_to_move[ii]#int(input("Please select the cell to move" + str(movement_actions_list) + ": "))
            ii += 1 #@fl

            avail_actions_mov = cell_objects[cell].get_actions()
            avail_neighbors = cell_objects[cell].get_neighbor()
            if np.any(expert_action_mov == np.array(avail_actions_mov)):
                new_state = avail_neighbors[np.where(expert_action_mov == np.array(avail_actions_mov))[0][0]]
            else:
                new_state = current_state
            new_cell = new_state
            uav.set_cell_id(cid=new_cell)
            uav.set_location(loc=cell_objects[new_cell].get_location())
            uav.set_hop(hop=uav.get_hop()+1)

            suggested_power = min(tx_powers) + (Number_of_neighbor_UEs.get('Max') -
                                                cell_objects[new_state].get_num_neighbor_ues()) / \
                              (Number_of_neighbor_UEs.get('Max') - Number_of_neighbor_UEs.get('Min')) * \
                              (max(tx_powers) - min(tx_powers))

            print("\n********** INFO:\n Number of Neighbor UEs: ", cell_objects[new_state].get_num_neighbor_ues(), '\n',
                  "Suggested Power: ", suggested_power)
            aa = [abs(pp[i] - suggested_power) for i in range(len(pp))]
            indx= aa.index(min(aa)) #@fl
            expert_action_power = pp[indx] #float(input("Please select the TX Power" + str(tx_powers) + ":"))
            expert_action = multi_actions_to_action(expert_action_mov, expert_action_power)
            uav.set_power(tr_power=expert_action_power)

            interference, sinr, throughput, interference_ues, max_throughput, = uav.uav_perform_task(cell_objects,
                                                                                                          ues_objects)

            print("\n********** INFO:\n",
                  "Episode: ", episode+1, '\n',
                  "Distance: ", distance + 1, '\n',
                  "Interference on UAV: ", interference, '\n',
                  "SINR: ", sinr, '\n',
                  "Throughput: ", throughput, '\n',
                  "Max Throughput: ", max_throughput, '\n',
                  "Interference on Neighbor UEs: ", interference_ues)
            features = get_features(state=new_cell, cell_objects=cell_objects, uav=uav, ues_objects=ues_objects)
            print("Features: ", features)
            expert_feature_expectation += get_feature_expectation(features, distance)
            print("Expert Feature Expectation: ", expert_feature_expectation)
            trajectory.append((current_state, expert_action, new_state, features, (interference, sinr, throughput,
                                                                                   interference_ues),
                               deepcopy(expert_feature_expectation)))
            arrow_patch_list = update_axes(ax_objects, prev_cell, cell_source, cell_destination, new_cell,
                                           expert_action_power, cell_objects[new_cell].get_location(),
                                           expert_action_mov, cell_objects[cell].get_location(), arrow_patch_list)

            plt.pause(0.001)
            prev_cell = new_cell
            if new_cell == cell_destination:
                done = True
            distance += 1
        trajectory.append(expert_feature_expectation)
        trajectories.append(trajectory)
        episode += 1
    if Config_Flags.get("SAVE_EXPERT_DATA"):
        file_name = '%d_Features_%d_trajectories_%d_length' % (num_features, num_trajectories, dist_limit)
        np.savez(ExpertPath + file_name, trajectories)


def get_features_draft(state, cell_objects, uav, ues_objects):
    phi_distance = 1 - np.power((cell_objects[state].get_distance()) / MAX_DISTANCE, 2.)
    phi_hop = 1 - np.power((uav.get_hop()) / dist_limit, 2.)
    # for neighbor in cell_objects[state].get_neighbor():
    #     num_neighbors_ues += len(cell_objects[neighbor].get_ues_idx())
    num_neighbors_ues = cell_objects[state].get_num_neighbor_ues()
    phi_ues = np.exp(-num_neighbors_ues/4 + 1)
    phi_throughput = np.power((uav.calc_throughput()) / uav.calc_max_throughput(cell_objects=cell_objects), 2)
    phi_interference = np.exp(-uav.calc_interference_ues(cells_objects=cell_objects, ues_objects=ues_objects))
    if num_features == 5:
        return phi_distance, phi_hop, phi_ues, phi_throughput, phi_interference
    else:  # In this case, the number of feature is 4 and we don't consider the hop count.
        return phi_distance, phi_ues, phi_throughput, phi_interference


def get_features_draft2(state, cell_objects, uav, ues_objects):
    phi_distance = 1 - np.power((cell_objects[state].get_distance()) / MAX_DISTANCE, 2.)
    phi_hop = 1 - np.power((uav.get_hop()) / dist_limit, 2.)
    num_neighbors_ues = cell_objects[state].get_num_neighbor_ues()
    phi_ues = 1 - np.power((num_neighbors_ues - MIN_UE_NEIGHBORS)/(MAX_UE_NEIGHBORS - MIN_UE_NEIGHBORS), 2)
    phi_throughput = np.power((uav.calc_throughput()) / uav.calc_max_throughput(cell_objects=cell_objects), 2)
    interference_on_ues = uav.calc_interference_ues(cells_objects=cell_objects, ues_objects=ues_objects)
    phi_interference = 1 - np.power((interference_on_ues - MIN_INTERFERENCE)/(MAX_INTERFERENCE - MIN_INTERFERENCE), 2)
    if num_features == 5:
        return phi_distance, phi_hop, phi_ues, phi_throughput, phi_interference
    else:  # In this case, the number of feature is 4 and we don't consider the hop count.
        return phi_distance, phi_ues, phi_throughput, phi_interference


def get_features(state, cell_objects, uav, ues_objects):
    phi_distance = np.power((cell_objects[state].get_distance()) / MAX_DISTANCE, 2.)
    phi_hop = 1 - np.power((uav.get_hop()) / dist_limit, 2.)
    num_neighbors_ues = cell_objects[state].get_num_neighbor_ues()
    phi_ues = np.power((num_neighbors_ues - MIN_UE_NEIGHBORS)/(MAX_UE_NEIGHBORS - MIN_UE_NEIGHBORS), 2)
    phi_throughput = np.power((uav.calc_throughput()) / uav.calc_max_throughput(cell_objects=cell_objects), 2)
    interference_on_ues = uav.calc_interference_ues(cells_objects=cell_objects, ues_objects=ues_objects)
    phi_interference = np.power((interference_on_ues - MIN_INTERFERENCE)/(MAX_INTERFERENCE - MIN_INTERFERENCE), 2)
    if state == cell_destination:
        phi_success = 5.0
    else:
        phi_success = 0.0

    if num_features == 5:
        return phi_distance, phi_success, phi_ues, phi_throughput, phi_interference
    else:  # In this case, the number of feature is 4 and we don't consider the hop count.
        return phi_success, phi_ues, phi_throughput, phi_interference


def get_feature_expectation(features, distance):
    return (gamma ** distance) * np.array(features)
