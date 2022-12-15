"""
#################################
# Shortest Path approach: e.x. Dijkstra
#################################
"""

#########################################################
# import libraries
import time
from random import seed
from random import randint
from datetime import datetime
from config import Config_Flags
import matplotlib.pyplot as plt
from config import Config_Power
from location import reset_axes
from location import update_axes
from config import Config_General
from config import Config_requirement
from config import Config_BehavioralCloning

#########################################################
# General Parameters
num_cells = Config_General.get('NUM_CELLS')
tx_powers = Config_Power.get('UAV_Tr_power')
dist_limit = Config_requirement.get('dist_limit')
NUM_EPISODES = Config_BehavioralCloning.get('NUM_TRAJECTORIES_EXPERT')

cell_source = 0
cell_destination = num_cells - 1

#########################################################
# Function definition


def short_path(uav, ues_objects, ax_objects, cell_objects):
    print(" ****** Mode: Shortest path with random power allocation policy by the drone ")
    seed(1732)
    prev_cell = 1
    episode = 0
    arrow_patch_list = []
    timer_start = time.perf_counter()
    print("......... TOTAL EPOCHS = ", NUM_EPISODES)
    while episode < NUM_EPISODES:
        distance = 0
        done = False
        arrow_patch_list = reset_axes(ax_objects=ax_objects, cell_source=cell_source,
                                      cell_destination=cell_destination, arrow_patch_list=arrow_patch_list)
        uav.uav_reset(cell_objects)
        while distance < dist_limit and not done:
            cell = uav.get_cell_id()
            avail_neighbors = cell_objects[cell].get_neighbor()
            avail_actions = cell_objects[cell].get_actions()

            # idx_rand = randint(0, len(avail_actions)-1)
            idx_rand = shortest_path_action(distance)
            action_rand = avail_actions[idx_rand]
            neighbor_rand = avail_neighbors[idx_rand]
            uav.set_action_movement(action=action_rand)
            uav.set_cell_id(cid=neighbor_rand)
            uav.set_location(loc=cell_objects[neighbor_rand].get_location())

            tx_index = randint(0, len(tx_powers)-1)
            tx_power = tx_powers[tx_index]
            uav.set_power(tr_power=tx_power)
            update_axes(ax_objects, prev_cell, cell_source, cell_destination, neighbor_rand, tx_power,
                        cell_objects[neighbor_rand].get_location(), action_rand, cell_objects[cell].get_location(),
                        arrow_patch_list)

            # interference = uav.calc_interference(cell_objects, ues_objects)
            # sinr, snr = uav.calc_sinr(cell_objects)
            # throughput = uav.calc_throughput()
            # interference_ues = uav.calc_interference_ues(cell_objects, ues_objects)

            # Should remove these above lines
            uav.uav_perform_task(cell_objects, ues_objects)
            if Config_Flags.get('Display_map'):
                plt.pause(0.001)
            prev_cell = neighbor_rand
            if neighbor_rand == cell_destination:
                done = True
            distance += 1

        episode += 1
        if episode % 200 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            timer_end = time.perf_counter()
            print(" ......... EPISODE = ", episode, "......... Current Time = ", current_time,
                  " ..... ELAPSED TIME = ", round(timer_end - timer_start, 2), " Seconds, ",
                  round((timer_end - timer_start) / 60, 2), " mins, ",
                  round((timer_end - timer_start) / 3600, 2), " hour")


def shortest_path_action(distance):
    index_movement_action = -1
    if distance == 0:
        index_movement_action = 1
    elif distance == 1:
        index_movement_action = 1
    elif distance == 2:
        index_movement_action = 1
    elif distance == 3:
        index_movement_action = 1
    elif distance == 4:
        index_movement_action = 0
    elif distance == 5:
        index_movement_action = 0

    if index_movement_action == -1:
        exit(' ........... Exit: wrong action, wrong cell')
    return index_movement_action
