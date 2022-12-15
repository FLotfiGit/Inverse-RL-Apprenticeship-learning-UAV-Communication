"""
#################################
# Evaluation of models and approaches
#################################
"""

#########################################################
# import libraries
import time
import pickle
import random
import numpy as np
from random import seed
from random import randint
from datetime import datetime
from config import Config_IRL
from config import Config_Path
from config import Config_Power
from config import Config_Flags
import matplotlib.pyplot as plt
from location import reset_axes
from location import update_axes
from config import Config_General
from config import Config_Evaluation
from inverserlSGD import get_features
from config import Config_requirement
from matplotlib.patches import Ellipse
from config import movement_actions_list
from utils import action_to_multi_actions
from utils import multi_actions_to_action
from inverserlSGD import get_greedy_action
from config import Config_BehavioralCloning
from shortestpath import shortest_path_action
from inverserlDQN import build_neural_network
from behavioral import train_model_behavioral
from inverserlDQN import get_greedy_action_dqn
from behavioral import load_expert_trajectories
from plotresults import plot_error_trajectories
from plotresults import plot_sample_trajectories
from plotresults import plot_training_trajectories

#########################################################
# General Parameters
WeightPath = Config_Path.get('WeightPath')
num_cells = Config_General.get('NUM_CELLS')
BCModelPath = Config_Path.get('BCModelPath')
tx_powers = Config_Power.get('UAV_Tr_power')
num_features = Config_IRL.get('NUM_FEATURES')
DQNModelPath = Config_Path.get('DQNModelPath')
SGDModelPath = Config_Path.get('SGDModelPath')
ResultPathPDF = Config_Path.get('ResultPathPDF')
ResultPathFIG = Config_Path.get('ResultPathFIG')
dist_limit = Config_requirement.get('dist_limit')
WeightPath_DQN = Config_Path.get('WeightPath_DQN')
NUM_TRAINING = Config_Evaluation.get('NUM_TRAINING')
NUM_EPISODES = Config_BehavioralCloning.get('NUM_TRAJECTORIES_EXPERT')

seed(1765)
cell_source = 0
action_list = []
cell_destination = num_cells - 1
for i in range(len(tx_powers) * len(movement_actions_list)):
    action_list.append(i)
action_array = np.array(action_list, dtype=np.int8)
#########################################################
# Function definition


def inverse_rl_hyper_distance():
    hyper_distance = [1.4808814477687855, 0.1947743531591942, 0.20178608693770217, 0.37417049598206165,
                      0.37853236096734755,
                      0.33176850191358415, 0.30923747765876247, 0.49682809809649375, 0.501600667872273,
                      0.05795836140829434, 0.13261291392221486, 0.17347945192419448]
    threshold = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    iteration_range = np.arange(0, len(hyper_distance))

    fig_hyper_distance = plt.figure(figsize=(8, 8))
    ax_hyper_distance = fig_hyper_distance.add_subplot(111)
    ax_hyper_distance.set_xlabel("Optimization Iteration", size=14, fontweight='bold')
    ax_hyper_distance.set_ylabel("Distance to expert feature distribution", size=14, fontweight='bold')

    ax_hyper_distance.plot(iteration_range, hyper_distance, color="black", linestyle='-', marker='o',
                           markersize='5', label='Hyper distance', linewidth=2)
    ax_hyper_distance.plot(iteration_range, threshold, color="blue", linestyle='--', label='Threshold',
                           linewidth=2)
    radius = 0.1
    ellipse = Ellipse((9, min(hyper_distance)), width=radius*max(iteration_range) / max(hyper_distance), height=radius,
                      color='r', alpha=0.5)
    ax_hyper_distance.add_artist(ellipse)
    ax_hyper_distance.grid(True)

    ax_hyper_distance.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
    file_fig_obj = "Hyper_distance.fig.pickle"
    file_fig_pdf = "Hyper_distance.pdf"

    if Config_Flags.get("SAVE_PLOT_PDF"):
        fig_hyper_distance.savefig(ResultPathPDF + file_fig_pdf, bbox_inches='tight')
    if Config_Flags.get("SAVE_PLOT_FIG"):
        pickle.dump(fig_hyper_distance, open(ResultPathFIG + file_fig_obj, 'wb'))


def evaluation_training(uav, ues_objects, ax_objects, cell_objects):
    weights_sgd, weights_norm_sgd, solution = load_weight_irl_sgd(iter_optimization=8) #(iter_optimization=0)
    trained_models_sgd = load_trained_model_sgd(learner_index=8)

    weights_dqn, weights_norm_dqn, solution_dqn = load_weight_irl_dqn(iter_optimization=1) #(iter_optimization=0)
    trained_models_dqn = load_trained_model_dqn(learner_index=1)
    trajectories_sgd_run = []
    trajectories_dqn_run = []
    num_run = 25
    for run in range(0, num_run):
        trajectories_sgd = run_training_sample(trained_models_sgd, uav, ues_objects, ax_objects,
                                               cell_objects, weights_norm_sgd, model_type="SGD")
        trajectories_sgd_run.append(trajectories_sgd)

        trajectories_dqn = run_training_sample(trained_models_dqn, uav, ues_objects, ax_objects,
                                                cell_objects, weights_norm_dqn, model_type="DQN")
        trajectories_dqn_run.append(trajectories_dqn)

    plot_training_trajectories(trajectories_sgd_run, trajectories_dqn_run, cell_objects)

 
def evaluation_scenario(uav, ues_objects, ax_objects, cell_objects):
    weights_sgd, weights_norm_sgd, solution = load_weight_irl_sgd(iter_optimization=8) #(iter_optimization=8)
    trained_models_sgd = load_trained_model_sgd(learner_index=8)
    trajectories_sgd = run_episode_sample(trained_models_sgd, uav, ues_objects, ax_objects,
                                           cell_objects, weights_norm_sgd, model_type="SGD")
    print(" ................... 1 Episode for Q-Learning using Linear Function Approximation ...................")

    weights_dqn, weights_norm_dqn, solution_dqn = load_weight_irl_dqn(iter_optimization=1) #(iter_optimization=3)
    trained_models_dqn = load_trained_model_dqn(learner_index=1)
    trajectories_dqn = run_episode_sample(trained_models_dqn, uav, ues_objects, ax_objects,
                                          cell_objects, weights_norm_dqn, model_type="DQN")
    print(" ................... 1 Episode for Deep Q-Network ...................")

    trajectories = load_expert_trajectories(uav, ues_objects, ax_objects, cell_objects, load_data=True)
    models_behavioral = train_model_behavioral(trajectories, load_model=True)
    weights_norm_bc = np.array([0, 0, 0, 0, 0])
    trajectories_bc = run_episode_sample(models_behavioral, uav, ues_objects, ax_objects,
                                          cell_objects, weights_norm_bc, model_type="BC")
    print(" ................... 1 Episode for Behavioral Cloning ...................")

    weights_norm_shortest = np.array([0, 0, 0, 0, 0])
    trajectories_shortest = run_episode_sample(None, uav, ues_objects, ax_objects,
                                               cell_objects, weights_norm_shortest, model_type="SHORT")
    print(" ................... 1 Episode for Shortest Path ...................")

    weights_norm_random = np.array([0, 0, 0, 0, 0])
    trajectories_random = run_episode_sample(None, uav, ues_objects, ax_objects,
                                               cell_objects, weights_norm_random, model_type="RND")
    print(" ................... 1 Episode for Random Behavior ...................")

    plot_sample_trajectories(trajectories_sgd, trajectories_dqn, trajectories_bc, trajectories_shortest,
                             trajectories_random, cell_objects)


def evaluation_error(uav, ues_objects, ax_objects, cell_objects):
    weights_sgd, weights_norm_sgd, solution = load_weight_irl_sgd(iter_optimization=8)#(iter_optimization=8)
    trained_models_sgd = load_trained_model_sgd(learner_index=8)
    trajectories_sgd = run_episode_sample(trained_models_sgd, uav, ues_objects, ax_objects,
                                          cell_objects, weights_norm_sgd, model_type="SGD", error=True)
    print(" ................... 1 Episode for Q-Learning using Linear Function Approximation ...................")

    weights_dqn, weights_norm_dqn, solution_dqn = load_weight_irl_dqn(iter_optimization=1)#(iter_optimization=3)
    trained_models_dqn = load_trained_model_dqn(learner_index=1)
    trajectories_dqn = run_episode_sample(trained_models_dqn, uav, ues_objects, ax_objects,
                                          cell_objects, weights_norm_dqn, model_type="DQN", error=True)
    print(" ................... 1 Episode for Deep Q-Network ...................")

    trajectories = load_expert_trajectories(uav, ues_objects, ax_objects, cell_objects, load_data=True)
    models_behavioral = train_model_behavioral(trajectories, load_model=True)
    weights_norm_bc = np.array([0, 0, 0, 0, 0])
    trajectories_bc = run_episode_sample(models_behavioral, uav, ues_objects, ax_objects,
                                         cell_objects, weights_norm_bc, model_type="BC", error=True)
    print(" ................... 1 Episode for Behavioral Cloning ...................")

    plot_error_trajectories(trajectories_sgd, trajectories_dqn, trajectories_bc, cell_objects)


def load_weight_irl_sgd(iter_optimization):
    #weight_file_name_np = 'weights_iter_%d_features_%d_epochs_%d.npz' % (iter_optimization, num_features, 10002)
    weight_file_name_np = 'weights_iter_%d_features_%d_epochs_%d.npz' % (iter_optimization, num_features, 10000)

    weight, weight_norm = np.load(WeightPath + weight_file_name_np).get('weight_list')[iter_optimization][0], \
                          np.load(WeightPath + weight_file_name_np).get('weight_list')[iter_optimization][1]
    return weight, weight_norm, None


def load_trained_model_sgd(learner_index):
    file_sgd_models_save = SGDModelPath + 'SGD_Feature_%d_learner_%d_index_EPOCHS_%d' % (num_features,
                                                                                         learner_index, 10002)
    with open(file_sgd_models_save, "rb") as file_obj:
        models = pickle.load(file_obj)
    return models


def load_weight_irl_dqn(iter_optimization):
    weight_file_name_np = 'weights_dqn_iter_%d_features_%d_epochs_%d.npz' % (iter_optimization, num_features, 10000)
    weight, weight_norm = np.load(WeightPath_DQN + weight_file_name_np).get('weight_list')[iter_optimization][0], \
                          np.load(WeightPath_DQN + weight_file_name_np).get('weight_list')[iter_optimization][1]
    return weight, weight_norm, None


def load_trained_model_dqn(learner_index):
    model = build_neural_network()
    file_dqn_models_save = DQNModelPath + 'DQN_Feature_%d_learner_%d_index_EPOCHS_%d.h5' % (num_features,
                                                                                            learner_index, 10000)
    model.load_weights(file_dqn_models_save)
    return model


def run_training_sample(models, uav, ues_objects, ax_objects, cell_objects, weights, model_type="SGD"):
    episode = 0
    epsilon_decay = 1
    trajectories = []
    arrow_patch_list = []
    prev_cell = 1
    timer_start = time.perf_counter()
    print("......... TOTAL RUNs = ", NUM_TRAINING)
    while episode < NUM_TRAINING:
        distance = 0
        done = False
        trajectory = []
        uav.uav_reset(cell_objects)
        arrow_patch_list = reset_axes(ax_objects=ax_objects, cell_source=cell_source, cell_destination=cell_destination,
                                      arrow_patch_list=arrow_patch_list)
        while distance < dist_limit and not done:
            current_cell = uav.get_cell_id()
            interference, sinr, throughput, interference_ues, max_throughput = uav.uav_perform_task(cell_objects,
                                                                                                    ues_objects)
            if Config_Flags.get('PRINT_INFO'):
                print("\n********** INFO:\n",
                      "Episode: ", episode + 1, '\n',
                      "Distance: ", distance, '\n',
                      "Current Cell:", current_cell, '\n',
                      "Current State \n",
                      "Interference on UAV: ", interference, '\n',
                      "SINR: ", sinr, '\n',
                      "Throughput: ", throughput, '\n',
                      "Interference on Neighbor UEs: ", interference_ues)
            features_current_state = get_features(cell=current_cell, cell_objects=cell_objects, uav=uav,
                                                  ues_objects=ues_objects)

            if random.random() < epsilon_decay:
                action = randint(0, len(action_list) - 1)
            else:
                if model_type == "SGD":
                    action = get_greedy_action(models, features_current_state, None)
                elif model_type == "DQN":
                    # Model Type is DQN
                    action = get_greedy_action_dqn(models, features_current_state)
                else:
                    # Model is wrong
                    action = None
                    exit(' ........... Exit: wrong model type')

            action_movement_index, action_tx_index = action_to_multi_actions(action)
            action_movement = action_movement_index + 1
            action_power = tx_powers[action_tx_index]

            # Calculate the next_state
            avail_actions_mov = cell_objects[current_cell].get_actions()
            avail_neighbors = cell_objects[current_cell].get_neighbor()
            if np.any(action_movement == np.array(avail_actions_mov)):
                new_cell = avail_neighbors[np.where(action_movement == np.array(avail_actions_mov))[0][0]]
            else:
                new_cell = current_cell

            uav.set_cell_id(cid=new_cell)
            uav.set_location(loc=cell_objects[new_cell].get_location())
            uav.set_hop(hop=uav.get_hop() + 1)
            uav.set_power(tr_power=action_power)

            interference_next, sinr_next, throughput_next, interference_ues_next, max_throughput_next = \
                uav.uav_perform_task(cell_objects, ues_objects)
            if Config_Flags.get('PRINT_INFO'):
                print("\n********** INFO:\n",
                      "Episode: ", episode + 1, '\n',
                      "Distance: ", distance + 1, '\n',
                      "New Cell:", new_cell, '\n',
                      "Next State \n",
                      "Action_power: ", action_power, '\n',
                      "Interference on UAV: ", interference_next, '\n',
                      "SINR: ", sinr_next, '\n',
                      "Throughput: ", throughput_next, '\n',
                      "Interference on Neighbor UEs: ", interference_ues_next)
            features_next_state = get_features(cell=new_cell, cell_objects=cell_objects, uav=uav,
                                               ues_objects=ues_objects)
            # learner_feature_expectation[episode, :] += get_feature_expectation(features_next_state, distance)
            # Calculate the reward
            immediate_reward = np.dot(weights, features_next_state)
            arrow_patch_list = update_axes(ax_objects, prev_cell, cell_source, cell_destination, new_cell,
                                           action_power, cell_objects[new_cell].get_location(),
                                           action_movement, cell_objects[current_cell].get_location(), arrow_patch_list)
            if Config_Flags.get('Display_map'):
                plt.pause(0.001)

            trajectory.append((features_current_state, (interference, sinr, throughput, interference_ues), action,
                               features_next_state, (interference_next, sinr_next, throughput_next,
                                                     interference_ues_next),
                               immediate_reward, new_cell))
            if new_cell == cell_destination:  # This is the termination point
                done = True
            prev_cell = new_cell
            distance += 1

        trajectories.append(trajectory)
        episode += 1
        arrow_patch_list = reset_axes(ax_objects=ax_objects, cell_source=cell_source, cell_destination=cell_destination,
                                      arrow_patch_list=arrow_patch_list)

        if epsilon_decay > 0 and episode:
            epsilon_decay -= (2 / NUM_TRAINING)
            if epsilon_decay < 0:
                epsilon_decay = 0

        if episode % int(NUM_TRAINING/50) == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            timer_end = time.perf_counter()
            print(" ......... EPISODE = ", episode, "......... Current Time = ", current_time,
                  " ..... ELAPSED TIME = ", round(timer_end - timer_start, 2), " Seconds, ",
                  round((timer_end - timer_start) / 60, 2), " mins, ",
                  round((timer_end - timer_start) / 3600, 2), " hour")
            print("epsilon_decay: ", epsilon_decay)

    return trajectories


def run_episode_sample(models, uav, ues_objects, ax_objects, cell_objects, weights, model_type="SGD", error=False):
    episode = 0
    trajectories = []
    arrow_patch_list = []
    prev_cell = 1
    num_samples = 1
    clf_xgb = None
    if model_type == "BC":
        clf_sgd, clf_svm, clf_tree, clf_gradient_boosting_classifier, clf_xgb = models[0], models[1], models[2], \
                                                                                models[3], models[4]
    print("......... TOTAL Samples = ", num_samples)
    while episode < num_samples:
        distance = 0
        done = False
        trajectory = []
        uav.uav_reset(cell_objects)
        arrow_patch_list = reset_axes(ax_objects=ax_objects, cell_source=cell_source, cell_destination=cell_destination,
                                      arrow_patch_list=arrow_patch_list)
        if error:
            uav.set_location(loc=cell_objects[5].get_location())
            uav.set_cell_id(cid=5)
        while distance < dist_limit and not done:
            current_cell = uav.get_cell_id()
            interference, sinr, throughput, interference_ues, max_throughput = uav.uav_perform_task(cell_objects,
                                                                                                    ues_objects)
            if Config_Flags.get('PRINT_INFO'):
                print("\n********** INFO:\n",
                      "Episode: ", episode + 1, '\n',
                      "Distance: ", distance, '\n',
                      "Current Cell:", current_cell, '\n',
                      "Current State \n",
                      "Interference on UAV: ", interference, '\n',
                      "SINR: ", sinr, '\n',
                      "Throughput: ", throughput, '\n',
                      "Interference on Neighbor UEs: ", interference_ues)
            features_current_state = get_features(cell=current_cell, cell_objects=cell_objects, uav=uav,
                                                  ues_objects=ues_objects)

            if model_type == "SGD":
                # Model Type is Q-learning with Linear function approximation
                action = get_greedy_action(models, features_current_state, None)
            elif model_type == "DQN":
                # Model Type is DQN
                action = get_greedy_action_dqn(models, features_current_state)
            elif model_type == "BC":
                # Model Type is Behavioral Cloning
                action = clf_xgb.predict(np.array(features_current_state).reshape(1, -1))
            elif model_type == "SHORT":
                # Model Type is Shortest Path
                idx_shortest = shortest_path_action(distance)
                action_mov = idx_shortest + 1
                tx_index = np.random.choice(np.arange(0, len(tx_powers)) - 1, p=[0.01, 0.01, 0.01, 0.02, 0.47, 0.48])
                tx_power = tx_powers[tx_index]
                action = multi_actions_to_action(action_mov, tx_power)
            elif model_type == "RND":
                # Model Type is Random
                avail_actions = cell_objects[current_cell].get_actions()
                idx_rand = randint(0, len(avail_actions) - 1)
                action_rand = avail_actions[idx_rand]
                tx_index = np.random.choice(np.arange(0, len(tx_powers)) - 1, p=[0.01, 0.01, 0.01, 0.02, 0.47, 0.48])
                tx_rand = tx_powers[tx_index]
                action = multi_actions_to_action(action_rand, tx_rand)
            else:
                # Model is wrong
                action = None
                exit(' ........... Exit: wrong model type')

            action_movement_index, action_tx_index = action_to_multi_actions(action)
            action_movement = action_movement_index + 1
            action_power = tx_powers[action_tx_index]

            # Calculate the next_state
            avail_actions_mov = cell_objects[current_cell].get_actions()
            avail_neighbors = cell_objects[current_cell].get_neighbor()
            if np.any(action_movement == np.array(avail_actions_mov)):
                new_cell = avail_neighbors[np.where(action_movement == np.array(avail_actions_mov))[0][0]]
            else:
                new_cell = current_cell

            uav.set_cell_id(cid=new_cell)
            uav.set_location(loc=cell_objects[new_cell].get_location())
            uav.set_hop(hop=uav.get_hop() + 1)
            uav.set_power(tr_power=action_power)

            interference_next, sinr_next, throughput_next, interference_ues_next, max_throughput_next = \
                uav.uav_perform_task(cell_objects, ues_objects)
            if Config_Flags.get('PRINT_INFO'):
                print("\n********** INFO:\n",
                      "Episode: ", episode + 1, '\n',
                      "Distance: ", distance + 1, '\n',
                      "New Cell:", new_cell, '\n',
                      "Next State \n",
                      "Action_power: ", action_power, '\n',
                      "Interference on UAV: ", interference_next, '\n',
                      "SINR: ", sinr_next, '\n',
                      "Throughput: ", throughput_next, '\n',
                      "Interference on Neighbor UEs: ", interference_ues_next)
            features_next_state = get_features(cell=new_cell, cell_objects=cell_objects, uav=uav,
                                               ues_objects=ues_objects)
            # learner_feature_expectation[episode, :] += get_feature_expectation(features_next_state, distance)
            # Calculate the reward
            immediate_reward = np.dot(weights, features_next_state)
            arrow_patch_list = update_axes(ax_objects, prev_cell, cell_source, cell_destination, new_cell,
                                           action_power, cell_objects[new_cell].get_location(),
                                           action_movement, cell_objects[current_cell].get_location(), arrow_patch_list)
            if Config_Flags.get('Display_map'):
                plt.pause(0.001)

            trajectory.append((features_current_state, (interference, sinr, throughput, interference_ues), action,
                               features_next_state, (interference_next, sinr_next, throughput_next,
                                                     interference_ues_next),
                               immediate_reward, new_cell))
            if new_cell == cell_destination:  # This is the termination point
                done = True
            prev_cell = new_cell
            distance += 1

        trajectories.append(trajectory)
        episode += 1
        arrow_patch_list = reset_axes(ax_objects=ax_objects, cell_source=cell_source, cell_destination=cell_destination,
                                      arrow_patch_list=arrow_patch_list)

    return trajectories
