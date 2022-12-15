"""
#################################
# Inverse Reinforcement Learning using Deep Q-Network
#################################
"""

#########################################################
# import libraries
import os
import time
import pickle
import random
import tensorflow
import numpy as np
from random import seed
from copy import deepcopy
from random import randint
from config import Config_IRL
from datetime import datetime
from config import Config_Path
from config import Config_Flags
from config import Config_Power
from location import reset_axes
import matplotlib.pyplot as plt
from location import update_axes
from config import Config_General
from config import Config_IRL_DQN
from tensorflow.keras import Input
from inverserlSGD import optimization
from inverserlSGD import get_features
from config import Config_requirement
from config import movement_actions_list
from utils import action_to_multi_actions
from plotresults import plot_reward_irl_dqn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from inverserlSGD import get_feature_expectation
from tensorflow.keras.layers import Dense, Dropout
from inverserlSGD import load_expert_feature_expectation

#########################################################
# General Parameters
NUM_PLAY = Config_IRL.get('NUM_PLAY')
LOAD_IRL = Config_Flags.get('LOAD_IRL')
ExpertPath = Config_Path.get('ExpertPath')
num_cells = Config_General.get('NUM_CELLS')
tx_powers = Config_Power.get('UAV_Tr_power')
BATCH_SIZE = Config_IRL_DQN.get('BATCH_SIZE')
num_features = Config_IRL.get('NUM_FEATURES')
NUM_EPOCHS = Config_IRL_DQN.get('NUM_EPOCHS')
INIT_LR = Config_IRL_DQN.get('LEARNING_RATE')
DQNModelPath = Config_Path.get('DQNModelPath')
gamma_discount = Config_IRL.get('GAMMA_DISCOUNT')
dist_limit = Config_requirement.get('dist_limit')
WeightPath_DQN = Config_Path.get('WeightPath_DQN')
BUFFER_LENGTH = Config_IRL_DQN.get('BUFFER_LENGTH')
epsilon_opt = Config_IRL.get('EPSILON_OPTIMIZATION')
InverseRLPathDQN = Config_Path.get('InverseRLPathDQN')
num_trajectories = Config_IRL.get('NUM_TRAJECTORIES_EXPERT')

seed(1369)
cell_source = 0
action_list = []
cell_destination = num_cells - 1
#####################################
# Disabling the GPU to test the speed
if Config_Flags.get('DISABLE_GPU'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#########################################
num_required_replays = int(NUM_EPOCHS / 10)
# num_required_replays = 1500
for i in range(len(tx_powers) * len(movement_actions_list)):
    action_list.append(i)
action_array = np.array(action_list, dtype=np.int8)
#########################################################
# Function definition


def inverse_rl_dqn(uav, ues_objects, ax_objects, cell_objects):
    weight_list = []
    solution_list = []
    hyper_dist = []
    trained_models = None
    iter_optimization = 0
    weight_file_name_txt = 'weights_dqn_features_%d_epochs_%d.txt' % (num_features, NUM_EPOCHS)
    weight_file = None
    if Config_Flags.get('SAVE_IRL_WEIGHT'):
        weight_file = open(WeightPath_DQN + weight_file_name_txt, 'w')
    expert_policy_feature_expectation = load_expert_feature_expectation()
    # expert_policy_feature_expectation = [Dist, 		  Success, 		UE, 		Throughput,  Interference]
    # expert_policy_feature_expectation(5) = [1.96955769, 4.9700749 , 0.29048563, 5.12332752, 0.31299007]
    # expert_policy_feature_expectation(4) = [4.9700749  0.29048563 5.12332752 0.31299007] # no Dist

    if num_features == 5:
        learner_policy_feature_expectation = [[1.96955769, 4.2700749, 0.49048563, 4.52332752, 0.51299007]]
    else:  # In this case, the number of feature is 4 and we don't consider the hop count(distance).
        learner_policy_feature_expectation = [[4.2700749, 0.49048563, 6.12332752, 0.61299007]]

    while True:
        if LOAD_IRL:
            weights, weights_norm, solution = load_weight_irl_dqn(iter_optimization)
        else:
            weights, weights_norm, solution = optimization(expert_policy_feature_expectation,
                                                           learner_policy_feature_expectation)
            print("Optimization status is: ", solution.get('status'))
            if Config_Flags.get('SAVE_IRL_WEIGHT'):
                weight_list.append((weights, weights_norm))
                solution_list.append(solution)
                weight_file.write(str(weight_list[-1]))
                weight_file_name_np = 'weights_dqn_iter_%d_features_%d_epochs_%d' % (iter_optimization, num_features,
                                                                                     NUM_EPOCHS)
                np.savez(WeightPath_DQN + weight_file_name_np, weight_list=weight_list, solution_list=solution_list)

        print("\nweights: ", weights, '\n', "weights_norm: ", weights_norm, '\n')
        model_type = "DQN"
        if not LOAD_IRL:
            trained_models = learner_dqn(weights_norm, uav, ues_objects, ax_objects, cell_objects,
                                                               iter_optimization)

        if LOAD_IRL:
            trained_models = load_trained_model_dqn(learner_index=iter_optimization)

        _, tested_policy_feature_expectation = run_trained_model(trained_models, uav, ues_objects, ax_objects,
                                                                 cell_objects, weights_norm, model_type=model_type)

        print("\ntested_policy_feature_expectation: ", tested_policy_feature_expectation)
        learner_policy_feature_expectation.append(tested_policy_feature_expectation.tolist())
        print("\nweights: ", weights, '\n', "weights_norm: ", weights_norm, '\n')
        hyper_distance = np.abs(np.dot(weights_norm, np.asarray(expert_policy_feature_expectation) -
                                       np.asarray(learner_policy_feature_expectation[-1])))
        print("...... Learner = ", iter_optimization, "  Hyper Distance = ", hyper_distance)
        hyper_dist.append(hyper_distance)
        if hyper_distance < epsilon_opt:
            # We are done with the Weight learning for the reward function and policy learning.
            # Now we have to Save the finalized weights for the reward function and also the learned policy for the
            # related weights.
            break
        else:
            # We have to find the weights again based on the updated learner_policy_feature_expectation. Going up to the
            # beginning of the loop
            pass
        iter_optimization += 1
    weight_file.close()
    np.savetxt('hyper_distance_vector.csv',hyper_dist)



def load_weight_irl_dqn(iter_optimization):
    weight_file_name_np = 'weights_dqn_iter_%d_features_%d_epochs_%d.npz' % (iter_optimization, num_features,
                                                                             NUM_EPOCHS)
    weight, weight_norm = np.load(WeightPath_DQN + weight_file_name_np).get('weight_list')[iter_optimization][0], \
                          np.load(WeightPath_DQN + weight_file_name_np).get('weight_list')[iter_optimization][1]
    return weight, weight_norm, None


def build_neural_network():
    input_dim = num_features
    model = Sequential()
    model.add(Input(shape=(input_dim, )))
    # First Layer
    model.add(Dense(units=30, activation='relu', kernel_initializer='lecun_uniform'))
    # model.add(Dropout(0.2))

    # Second Layer
    model.add(Dense(units=30, activation='relu', kernel_initializer='lecun_uniform'))
    # model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(units=len(action_list), activation='linear', kernel_initializer='lecun_uniform'))
    opt = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
    model.compile(optimizer=opt, loss='mse', metrics=["accuracy"])
    return model


def learner_dqn(weights, uav, ues_objects, ax_objects, cell_objects, learner_index):
    episode = 0
    trajectories = []
    arrow_patch_list = []
    epsilon_decay = 1
    prev_cell = 1
    model = build_neural_network()
    timer_start = time.perf_counter()
    print("......... TOTAL EPOCHS = ", NUM_EPOCHS)
    replay = []  # tuples of (S, A, R, S').
    while episode < NUM_EPOCHS:
        trajectory = []
        distance = 0
        done = False
        uav.uav_reset(cell_objects)
        arrow_patch_list = reset_axes(ax_objects=ax_objects, cell_source=cell_source, cell_destination=cell_destination,
                                      arrow_patch_list=arrow_patch_list)
        learner_feature_expectation = np.zeros(num_features, dtype=float)
        while distance < dist_limit and not done:
            current_cell = uav.get_cell_id()
            # Calculate the current state
            interference, sinr, throughput, interference_ues, max_throughput = uav.uav_perform_task(cell_objects,
                                                                                                    ues_objects)
            features_current_state = get_features(cell=current_cell, cell_objects=cell_objects, uav=uav,
                                                  ues_objects=ues_objects)
            if Config_Flags.get('PRINT_INFO'):
                print("\n********** INFO:\n",
                      "Episode: ", episode + 1, '\n',
                      "Distance: ", distance, '\n',
                      "Current Cell:", current_cell, '\n',
                      "Current State \n",
                      "Interference on UAV: ", interference, '\n',
                      "SINR: ", sinr, '\n',
                      "Throughput: ", throughput, '\n',
                      "Max Throughput: ", max_throughput, '\n',
                      "Interference on Neighbor UEs: ", interference_ues, '\n',
                      "features_current_state: ", features_current_state)
            if random.random() < epsilon_decay or episode < num_required_replays:
                action = randint(0, len(action_list)-1)
            else:
                # Bring the model here for the greedy action
                action = get_greedy_action_dqn(model, features_current_state)

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
            features_next_state = get_features(cell=new_cell, cell_objects=cell_objects, uav=uav,
                                               ues_objects=ues_objects)

            if Config_Flags.get('PRINT_INFO'):
                print("\n********** INFO:\n",
                      "Episode: ", episode + 1, '\n',
                      "Distance: ", distance + 1, '\n',
                      "New Cell:", new_cell, '\n',
                      "Next State \n",
                      "Interference on UAV: ", interference_next, '\n',
                      "SINR: ", sinr_next, '\n',
                      "Throughput: ", throughput_next, '\n',
                      "Max Throughput: ", max_throughput_next, '\n',
                      "Interference on Neighbor UEs: ", interference_ues_next, '\n',
                      "features_next_state: ", features_next_state)

            learner_feature_expectation += get_feature_expectation(features_next_state, distance)

            immediate_reward = np.dot(weights, features_next_state)

            replay.append((features_current_state, action, immediate_reward, features_next_state, new_cell))

            arrow_patch_list = update_axes(ax_objects, prev_cell, cell_source, cell_destination, new_cell,
                                           action_power, cell_objects[new_cell].get_location(),
                                           action_movement, cell_objects[current_cell].get_location(), arrow_patch_list)

            trajectory.append((features_current_state, (interference, sinr, throughput, interference_ues), action,
                               features_next_state, (interference_next, sinr_next, throughput_next,
                                                     interference_ues_next),
                               immediate_reward, deepcopy(learner_feature_expectation)))

            if new_cell == cell_destination:  # This is the termination point
                done = True

            # *****************************************************************************************
            # Train the Deep Q Network: Deep Reinforcement Learning
            # if episode > num_required_replays:
            #     if len(replay) > BUFFER_LENGTH:
            #         replay.pop(0)
            #
            #     batch = random.sample(replay, BATCH_SIZE)
            #     x_train, y_train = get_batch_ready(batch, model)
            #     model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, verbose=2)
            # *****************************************************************************************
            if Config_Flags.get('Display_map'):
                plt.pause(0.01)
            prev_cell = new_cell
            distance += 1

        # *****************************************************************************************
        # Train the Deep Q Network: Deep Reinforcement Learning
        if episode > num_required_replays:
            if len(replay) > BUFFER_LENGTH:
                replay.pop(0)

            batch = random.sample(replay, BATCH_SIZE)
            x_train, y_train = get_batch_ready(batch, model)
            model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, verbose=0)
        # *****************************************************************************************

        if epsilon_decay > 0.005 and episode > num_required_replays:
            epsilon_decay -= (2 / NUM_EPOCHS)

        trajectory.append(learner_feature_expectation)
        trajectories.append(trajectory)

        episode += 1
        if episode % 100 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            timer_end = time.perf_counter()
            print(" ......... EPISODE = ", episode, "......... Current Time = ", current_time,
                  " ..... ELAPSED TIME = ", round(timer_end - timer_start, 2), " Seconds, ",
                  round((timer_end - timer_start) / 60, 2), " mins, ",
                  round((timer_end - timer_start) / 3600, 2), " hour")

    trajectories.append(model)
    trajectories.append(learner_index)

    if Config_Flags.get("PLOT_RESULTS"):
        plot_reward_irl_dqn(trajectories, learner_index)

    # I have to save the trajectories' information on numpy files (Drive) for later evaluation
    if Config_Flags.get("SAVE_IRL_DATA_DQN"):
        learner_irl_dqn_file_name_np = 'DQN_Feature_%d_learner_%d_index_EPOCHS_%d' % (num_features, learner_index,
                                                                                      NUM_EPOCHS)
        np.savez(InverseRLPathDQN + learner_irl_dqn_file_name_np, trajectories=trajectories)

    if Config_Flags.get('SAVE_MODEL_IRL_DQN'):
        file_dqn_models_save = DQNModelPath + 'DQN_Feature_%d_learner_%d_index_EPOCHS_%d' % (num_features,
                                                                                             learner_index, NUM_EPOCHS)
        # pickle.dump(model, open(file_dqn_models_save, 'wb'))
        model.save_weights(file_dqn_models_save + ".h5", overwrite=True)
        print("Saving model %s - %d" % (file_dqn_models_save, learner_index))

    return model


def load_trained_model_dqn(learner_index):
    model = build_neural_network()
    file_dqn_models_save = DQNModelPath + 'DQN_Feature_%d_learner_%d_index_EPOCHS_%d.h5' % (num_features,
                                                                                            learner_index, NUM_EPOCHS)
    model.load_weights(file_dqn_models_save)
    return model


def learner_dqn_unlimited_distance(weights, uav, ues_objects, ax_objects, cell_objects,
                                                               iter_optimization):
    dist_infinite = 10000
    model = build_neural_network()

    return model


def get_greedy_action_dqn(model_dqn, features_state):
    q_value = model_dqn.predict(np.array(features_state).reshape(1, num_features), batch_size=1)
    action = np.argmax(q_value)
    return action


def get_batch_ready(batch, model):
    x_train = []
    y_train = []
    for memory in batch:
        # memory = (features_current_state, action, immediate_reward, features_next_state, new_cell)
        current_feature, action, reward, next_feature, next_cell = memory
        q_value_current = model.predict(np.array(current_feature).reshape(1, num_features), batch_size=1)
        q_value_next = model.predict(np.array(next_feature).reshape(1, num_features), batch_size=1)
        y = np.zeros((1, len(action_list)))
        y[:] = q_value_current[:]

        if next_cell == cell_destination:  # This is the termination point
            q_dqn_target = reward
        else:
            q_dqn_target = reward + (gamma_discount * np.max(q_value_next))

        y[0][action] = q_dqn_target
        x_train.append(np.array(current_feature).reshape(num_features, ))
        y_train.append(y.reshape(len(action_list), ))
    x_train_np = np.array(x_train)
    y_train_np = np.array(y_train)

    return x_train_np, y_train_np


def run_trained_model(models, uav, ues_objects, ax_objects, cell_objects, weights, model_type="DQN"):
    episode = 0
    trajectories = []
    arrow_patch_list = []
    prev_cell = 1
    print("......... TOTAL RUNs = ", NUM_PLAY)
    learner_feature_expectation = np.zeros((NUM_PLAY, num_features), dtype=float)
    while episode < NUM_PLAY:
        trajectory = []
        distance = 0
        done = False
        uav.uav_reset(cell_objects)
        arrow_patch_list = reset_axes(ax_objects=ax_objects, cell_source=cell_source, cell_destination=cell_destination,
                                      arrow_patch_list=arrow_patch_list)
        while distance < dist_limit and not done:
            current_cell = uav.get_cell_id()
            interference, sinr, throughput, interference_ues, max_throughput = uav.uav_perform_task(cell_objects,
                                                                                                    ues_objects)
            # if Config_Flags.get('PRINT_INFO'):
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

            if model_type == "DQN":
                action = get_greedy_action_dqn(models, features_current_state)
            else:
                # Model Type is SGD
                # action = get_greedy_action_dqn(models, features_current_state)
                action = None
                pass

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
            # if Config_Flags.get('PRINT_INFO'):
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
            learner_feature_expectation[episode, :] += get_feature_expectation(features_next_state, distance)
            # Calculate the reward
            immediate_reward = np.dot(weights, features_next_state)
            arrow_patch_list = update_axes(ax_objects, prev_cell, cell_source, cell_destination, new_cell,
                                           action_power, cell_objects[new_cell].get_location(),
                                           action_movement, cell_objects[current_cell].get_location(), arrow_patch_list)

            trajectory.append((features_current_state, (interference, sinr, throughput, interference_ues), action,
                               features_next_state, (interference_next, sinr_next, throughput_next,
                                                     interference_ues_next),
                               immediate_reward, deepcopy(learner_feature_expectation)))
            if new_cell == cell_destination:  # This is the termination point
                done = True
            prev_cell = new_cell
            distance += 1
        trajectory.append(learner_feature_expectation)
        trajectories.append(trajectory)
        episode += 1
        arrow_patch_list = reset_axes(ax_objects=ax_objects, cell_source=cell_source, cell_destination=cell_destination,
                                      arrow_patch_list=arrow_patch_list)

    return learner_feature_expectation, np.mean(learner_feature_expectation, axis=0)
