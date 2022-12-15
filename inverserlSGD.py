"""
#################################
# Inverse Reinforcement Learning using Stochastic Gradient Descent
#################################
"""

#########################################################
# import libraries
import time
import pickle
import random
import numpy as np
from config import Mode
from random import seed
from cvxopt import matrix
from copy import deepcopy
from cvxopt import solvers
from random import randint
from config import Config_IRL
from datetime import datetime
from config import Config_Path
from config import Config_Power
from config import Config_Flags
from location import reset_axes
import matplotlib.pyplot as plt
from location import update_axes
from config import Config_General
from config import Config_requirement
from config import movement_actions_list
from utils import action_to_multi_actions
from plotresults import plot_reward_irl_sgd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


#########################################################
# General Parameters
NUM_PLAY = Config_IRL.get('NUM_PLAY')
LOAD_IRL = Config_Flags.get('LOAD_IRL')
NUM_EPOCHS = Config_IRL.get('NUM_EPOCHS')
ExpertPath = Config_Path.get('ExpertPath')
WeightPath = Config_Path.get('WeightPath')
num_cells = Config_General.get('NUM_CELLS')
num_states = Config_General.get('NUM_CELLS')
tx_powers = Config_Power.get('UAV_Tr_power')
num_features = Config_IRL.get('NUM_FEATURES')
epsilon_grd = Config_IRL.get('EPSILON_GREEDY')
SGDModelPath = Config_Path.get('SGDModelPath')
InverseRLPath = Config_Path.get('InverseRLPath')
gamma_features = Config_IRL.get('GAMMA_FEATURES')
gamma_discount = Config_IRL.get('GAMMA_DISCOUNT')
dist_limit = Config_requirement.get('dist_limit')
epsilon_opt = Config_IRL.get('EPSILON_OPTIMIZATION')
MAX_DISTANCE = Config_requirement.get('MAX_DISTANCE')
num_trajectories = Config_IRL.get('NUM_TRAJECTORIES_EXPERT')
MIN_UE_NEIGHBORS = Config_requirement.get('MIN_UE_NEIGHBORS')
MAX_UE_NEIGHBORS = Config_requirement.get('MAX_UE_NEIGHBORS')
MIN_INTERFERENCE = Config_requirement.get('MIN_INTERFERENCE')
MAX_INTERFERENCE = Config_requirement.get('MAX_INTERFERENCE')

seed(1369)
cell_source = 0
action_list = []
cell_destination = num_cells - 1
num_required_replays = int(NUM_EPOCHS / 10)
# num_required_replays = 1500
for i in range(len(tx_powers) * len(movement_actions_list)):
    action_list.append(i)
action_array = np.array(action_list, dtype=np.int8)

#########################################################
# Function definition


def inverse_rl_sgd(uav, ues_objects, ax_objects, cell_objects):
    model_type = None
    weight_list = []
    solution_list = []
    trained_models = None
    iter_optimization = 0
    weight_file_name_txt = 'weights_features_%d_epochs_%d.txt' % (num_features, NUM_EPOCHS)
    weight_file = None
    if Config_Flags.get('SAVE_IRL_WEIGHT'):
        weight_file = open(WeightPath + weight_file_name_txt, 'w')

    expert_policy_feature_expectation = load_expert_feature_expectation()
    # expert_policy_feature_expectation = [Dist, 		  Success, 		UE, 		Throughput,  Interference]
    # expert_policy_feature_expectation(5) = [1.96955769, 4.9700749 , 0.29048563, 5.12332752, 0.31299007]
    # expert_policy_feature_expectation(4) = [4.9700749  0.29048563 5.12332752 0.31299007] # no Dist

    # Just some random feature expectation for the learner:
    if num_features == 5:
        learner_policy_feature_expectation = [[1.96955769, 4.2700749, 0.49048563, 4.52332752, 0.51299007]]
    else:  # In this case, the number of feature is 4 and we don't consider the hop count(distance).
        learner_policy_feature_expectation = [[4.2700749, 0.49048563, 6.12332752, 0.61299007]]

    random_initial_t = np.linalg.norm(expert_policy_feature_expectation -
                                      np.array(learner_policy_feature_expectation[0]))

    while True:
        if LOAD_IRL:
            weights, weights_norm, solution = load_weight_irl(iter_optimization)
        else:
            weights, weights_norm, solution = optimization(expert_policy_feature_expectation,
                                                           learner_policy_feature_expectation)
            print("Optimization status is: ", solution.get('status'))
            if solution.get('status') == "optimal":
                weight_list.append((weights, weights_norm))
                solution_list.append(solution)
                if Config_Flags.get('SAVE_IRL_WEIGHT'):
                    weight_file.write(str(weight_list[-1]))
                    weight_file_name_np = 'weights_iter_%d_features_%d_epochs_%d' % (iter_optimization, num_features,
                                                                                     NUM_EPOCHS)
                    np.savez(WeightPath + weight_file_name_np, weight_list=weight_list, solution_list=solution_list)

        print("\nweights: ", weights, '\n', "weights_norm: ", weights_norm, '\n')
        #  (A): Run another simulation based on the new weights to update the learner policy
        #  (Feature expectation policy) to run another simulation we can have simple Q learning model or
        #  a deep reinforcement learning one
        if not LOAD_IRL:
            # model = build_neural_network()
            if Mode == "IRL_DQN":
                model_type = "DQN"
                # trained_models = learner_dqn(model, weights_norm)
                pass
            if Mode == "IRL_SGD":
                model_type = "SGD"
                # [5.00947727 4.9700749  0.29048563 5.12332752 0.31299007]
                # weights_norm = np.asarray([6.80128574e-06,  0.86679868, -9.62677292e-01, 2.55648174e-02, -0.42602828])
                trained_models = learner_lfa_ql(weights_norm, uav, ues_objects, ax_objects, cell_objects,
                                                iter_optimization)
                # trained_models = learner_lfa_ql_unlimited_dist(weights_norm, uav, ues_objects, ax_objects,
                #                                                cell_objects,ziter_optimization)

        #  Update the learner policy (Feature expectation policy) and calculate the hyper distance between the
        #  current learner policy (Feature expectation policy) and the expert policy (Feature expectation policy).

        if LOAD_IRL:
            model_type = "SGD"
            trained_models = load_trained_model(learner_index=iter_optimization)
        # Another model_Type is "DQN"
        _, tested_policy_feature_expectation = run_trained_model(trained_models, uav, ues_objects, ax_objects,
                                                                 cell_objects, weights_norm, model_type=model_type)
        print("\ntested_policy_feature_expectation: ", tested_policy_feature_expectation)
        learner_policy_feature_expectation.append(tested_policy_feature_expectation.tolist())
        print("\nweights: ", weights, '\n', "weights_norm: ", weights_norm, '\n')
        hyper_distance = np.abs(np.dot(weights_norm, np.asarray(expert_policy_feature_expectation) -
                                       np.asarray(learner_policy_feature_expectation[-1])))
        print("...... Learner = ", iter_optimization, "  Hyper Distance = ", hyper_distance)

        # If the distance is less than a threshold, then break the optimization and report the optimal weights
        # and the optimal policy based on the imported weights else go to (A)
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

    if Config_Flags.get('SAVE_IRL_WEIGHT'):
        weight_file.close()


def load_expert_feature_expectation():
    file_name = '%d_Features_%d_trajectories_%d_length.npz' % (num_features, num_trajectories, dist_limit)
    readfile = np.load(ExpertPath + file_name, allow_pickle=True)
    # 'arr_0' = readfile.files[0]
    trajectories = readfile["arr_0"]
    sum_expert_feature_expectation = 0
    for trajectory in trajectories:
        sum_expert_feature_expectation += trajectory[-1]
    expert_feature_expectation = sum_expert_feature_expectation / num_trajectories
    # if num_features == 5:
    return expert_feature_expectation
    # else:
    #     return expert_feature_expectation[1:]
    # else:  # In this case, the number of feature is 4 and we don't consider the hop count.
    #     return np.concatenate((expert_feature_expectation[0], expert_feature_expectation[2:]), axis=None)
    # return np.delete(expert_feature_expectation, 1, axis=0)


def optimization(policy_expert, policies_agent):
    # https://cvxopt.org/examples/tutorial/qp.html
    # https://cvxopt.org/userguide/coneprog.html#quadratic-programming
    """
    To have more information about this Quadratic programming please read these topics:

    :param policy_expert:
    :param policies_agent:
    :return:
    """
    length = len(policy_expert)
    p = matrix(2.0 * np.eye(length), tc='d')
    q = matrix(np.zeros(length), tc='d')
    status = None
    removed_policy = []
    solution_weights = None
    # policy_subject = None
    # if num_features == 5:
    #     policy_subject = [np.array([-1., -1., -1., -1., -1.], dtype=float)]
    # if num_features == 4:
    #     policy_subject = [np.array([-1., -1., -1., -1.], dtype=float)]
    # policy_subject.append(policy_expert)
    policy_subject = [policy_expert]
    h_subject = [1]
    for policy in policies_agent:
        policy_subject.append(policy)
        h_subject.append(1)

    while status != "optimal":
        policy_subject_mat = np.array(policy_subject)
        policy_subject_mat[0] = -1 * policy_subject_mat[0]

        g = matrix(policy_subject_mat, tc='d')
        h = matrix(-np.array(h_subject), tc='d')
        solution_weights = solvers.qp(p, q, g, h)#, solver='mosek')  # @fl
        status = solution_weights['status']

        if status != "optimal":
            removed_policy.append(policy_subject.pop(1))
            h_subject.pop()
            print(" ********* Pop and removing the oldest policy to have optimal solution")

    if status == "optimal":
        weights = np.squeeze(np.asarray(solution_weights['x']))
        weights_normalized = weights / np.linalg.norm(weights)
        return weights, weights_normalized, solution_weights
    else:
        exit(' ........... Exit: Could not find the optimal solution')
        return None, None, solution_weights


def learner_lfa_ql(weights, uav, ues_objects, ax_objects, cell_objects, learner_index):
    # Q learning with Linear Function Approximation
    std_scale = StandardScaler()  # we should use partial_fit
    episode = 0
    trajectories = []
    arrow_patch_list = []
    epsilon_decay = 1
    prev_cell = 1
    sgd_models, std_scale = create_sgd_models(num_actions=len(action_list), std_scale=std_scale)
    timer_start = time.perf_counter()
    print("......... TOTAL EPOCHS = ", NUM_EPOCHS)
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

            # features_current_state = phi_distance, phi_hop, phi_ues, phi_throughput, phi_interference

            # Choose an action based on epsilon-greedy
            # if random.random() < epsilon_grd:
            if random.random() < epsilon_decay:
                action = randint(0, len(action_list)-1)
            else:
                # Bring the model here for the greedy action
                action = get_greedy_action(sgd_models, features_current_state, std_scale)
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

            # Calculate the reward
            immediate_reward = np.dot(weights, features_next_state)

            # Update the Next Q value and Calculate the td target
            q_value_next = sgd_predictor(sgd_models, features_next_state, std_scale)
            if new_cell == cell_destination:  # This is the termination point
                done = True
                q_td_target = immediate_reward
            else:
                q_td_target = immediate_reward + (gamma_discount * np.max(q_value_next))

            # Update the estimator(model)
            sgd_models, std_scale = update_sgd_models(sgd_models, features_current_state, action, q_td_target,
                                                      std_scale)

            arrow_patch_list = update_axes(ax_objects, prev_cell, cell_source, cell_destination, new_cell,
                                           action_power, cell_objects[new_cell].get_location(),
                                           action_movement, cell_objects[current_cell].get_location(), arrow_patch_list)
            trajectory.append((features_current_state, (interference, sinr, throughput, interference_ues), action,
                               features_next_state, (interference_next, sinr_next, throughput_next,
                                                     interference_ues_next),
                               immediate_reward, deepcopy(learner_feature_expectation)))
            if Config_Flags.get('Display_map'):
                plt.pause(0.01)
            prev_cell = new_cell
            distance += 1

        if epsilon_decay > 0.005 and episode > num_required_replays:
            epsilon_decay -= (2 / NUM_EPOCHS)

        # if epsilon_decay > 0.1 and episode > num_required_replays:
        #     epsilon_decay -= (1 / NUM_EPOCHS)

        trajectory.append(learner_feature_expectation)
        trajectories.append(trajectory)
        episode += 1
        if episode % 200 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            timer_end = time.perf_counter()
            print(" ......... EPISODE = ", episode, "......... Current Time = ", current_time,
                  " ..... ELAPSED TIME = ", round(timer_end - timer_start, 2), " Seconds, ",
                  round((timer_end - timer_start) / 60, 2), " mins, ",
                  round((timer_end - timer_start) / 3600, 2), " hour")
    trajectories.append(sgd_models)
    trajectories.append(learner_index)

    # I have to plot the reward behavior in one simulation to see how they have improvement and convergence.
    if Config_Flags.get("PLOT_RESULTS"):
        plot_reward_irl_sgd(trajectories, learner_index)

    # I have to save the trajectories' information on numpy files (Drive) for later evaluation
    if Config_Flags.get("SAVE_IRL_DATA"):
        learner_irl_file_name_np = 'Feature_%d_learner_%d_index_EPOCHS_%d' % (num_features, learner_index, NUM_EPOCHS)
        np.savez(InverseRLPath + learner_irl_file_name_np, trajectories=trajectories)

    # Let's save the SGD models for later
    if Config_Flags.get('SAVE_MODEL_IRL_SGD'):
        file_sgd_models_save = SGDModelPath + 'SGD_Feature_%d_learner_%d_index_EPOCHS_%d' % (num_features,
                                                                                             learner_index, NUM_EPOCHS)
        pickle.dump(sgd_models, open(file_sgd_models_save, 'wb'))

    return sgd_models


def get_features_draft(cell, cell_objects, uav, ues_objects):
    phi_distance = 1 - np.power((cell_objects[cell].get_distance()) / MAX_DISTANCE, 2.)
    phi_hop = 1 - np.power((uav.get_hop()) / dist_limit, 2.)
    num_neighbors_ues = cell_objects[cell].get_num_neighbor_ues()
    phi_ues = 1 - np.power((num_neighbors_ues - MIN_UE_NEIGHBORS) / (MAX_UE_NEIGHBORS - MIN_UE_NEIGHBORS), 2)
    phi_throughput = np.power((uav.calc_throughput()) / uav.calc_max_throughput(cell_objects=cell_objects), 2)
    interference_on_ues = uav.calc_interference_ues(cells_objects=cell_objects, ues_objects=ues_objects)
    phi_interference = 1 - np.power((interference_on_ues - MIN_INTERFERENCE) / (MAX_INTERFERENCE - MIN_INTERFERENCE), 2)
    if num_features == 5:
        return phi_distance, phi_hop, phi_ues, phi_throughput, phi_interference
    else:  # In this case, the number of feature is 4 and we don't consider the hop count.
        return phi_distance, phi_ues, phi_throughput, phi_interference


def get_features(cell, cell_objects, uav, ues_objects):
    phi_distance = np.power((cell_objects[cell].get_distance()) / MAX_DISTANCE, 2.)
    # phi_hop = 1 - np.power((uav.get_hop()) / dist_limit, 2.)
    num_neighbors_ues = cell_objects[cell].get_num_neighbor_ues()
    phi_ues = np.power((num_neighbors_ues - MIN_UE_NEIGHBORS)/(MAX_UE_NEIGHBORS - MIN_UE_NEIGHBORS), 2)
    phi_throughput = np.power((uav.calc_throughput()) / uav.calc_max_throughput(cell_objects=cell_objects), 2)
    interference_on_ues = uav.calc_interference_ues(cells_objects=cell_objects, ues_objects=ues_objects)
    phi_interference = np.power((interference_on_ues - MIN_INTERFERENCE)/(MAX_INTERFERENCE - MIN_INTERFERENCE), 2)
    if cell == cell_destination:
        phi_success = 5.0
    else:
        phi_success = 0.0

    if num_features == 5:
        return phi_distance, phi_success, phi_ues, phi_throughput, phi_interference
    else:  # In this case, the number of feature is 4 and we don't consider the hop count.
        return phi_success, phi_ues, phi_throughput, phi_interference


def get_feature_expectation(features, distance):
    return (gamma_features ** distance) * np.array(features)


def create_sgd_models(num_actions, std_scale):
    models = []
    #  Here after creating each model, we have to do partial fit with some initial values, otherwise we will face
    #  some errors because we are doing the first predict before the first update. If we don't do that, we probably
    #  get some errors.
    if num_features == 5:
        initial_values_features = \
            np.array([0.6944444444444445, 0.0, 0.0256, 0.6280412430050324, 0.03287622045477716]).reshape(1, -1)
    else:  # In this case, the number of feature is 4 and we don't consider the hop count.
        initial_values_features = \
            np.array([0.0, 0.0256, 0.6280412430050324, 0.03287622045477716]).reshape(1, -1)

    # std_scale.partial_fit(initial_values_features)
    # initial_values_features_scaled = std_scale.transform(initial_values_features)
    initial_values_features_scaled = initial_values_features
    # These are the initial feature values for the first state when the UAV is at location (x=0, y=0).
    for _ in range(0, num_actions):
        model = SGDRegressor(learning_rate="constant")
        model.partial_fit(initial_values_features_scaled, [0])
        models.append(model)
    return models, std_scale


def update_sgd_models(sgd_models, features_state, action, target, std_scale):
    std_scale.partial_fit(np.array(features_state).reshape(1, -1))
    # features_state_scaled = std_scale.transform(np.array(features_state).reshape(1, -1))
    features_state_scaled = np.array(features_state).reshape(1, -1)
    if Config_Flags.get('PRINT_INFO'):
        print("features_state = ", features_state, '\n'
              "features_state_scaled = ", features_state_scaled)
    sgd_models[action].partial_fit(features_state_scaled, [target])
    return sgd_models, std_scale


def sgd_predictor(sgd_models, features_state, std_scale):
    # features_state_scaled = std_scale.transform(np.array(features_state).reshape(1, -1))
    features_state_scaled = np.array(features_state).reshape(1, -1)
    return np.array([m.predict(features_state_scaled)[0] for m in sgd_models])


def get_greedy_action(sgd_models, features_state, std_scale):
    action_q_values = sgd_predictor(sgd_models, features_state, std_scale)
    action = np.argmax(action_q_values)
    return action


def run_trained_model(models, uav, ues_objects, ax_objects, cell_objects, weights, model_type="SGD"):
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

            if model_type == "SGD":
                action = get_greedy_action(models, features_current_state, None)
            else:
                # Model Type is DQN
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


def load_trained_model(learner_index):
    file_sgd_models_save = SGDModelPath + 'SGD_Feature_%d_learner_%d_index_EPOCHS_%d' % (num_features,
                                                                                         learner_index, NUM_EPOCHS)
    with open(file_sgd_models_save, "rb") as file_obj:
        models = pickle.load(file_obj)
    return models


def load_weight_irl(iter_optimization):
    weight_file_name_np = 'weights_iter_%d_features_%d_epochs_%d.npz' % (iter_optimization, num_features, NUM_EPOCHS)
    weight, weight_norm = np.load(WeightPath + weight_file_name_np).get('weight_list')[iter_optimization][0], \
                          np.load(WeightPath + weight_file_name_np).get('weight_list')[iter_optimization][1]
    return weight, weight_norm, None


def learner_lfa_ql_unlimited_dist(weights, uav, ues_objects, ax_objects, cell_objects, learner_index):
    # Q learning with Linear Function Approximation
    std_scale = StandardScaler()  # we should use partial_fit
    episode = 0
    dist_infinite = 10000
    trajectories = []
    arrow_patch_list = []
    epsilon_decay = 1
    prev_cell = 1
    sgd_models, std_scale = create_sgd_models(num_actions=len(action_list), std_scale=std_scale)
    timer_start = time.perf_counter()
    print("......... TOTAL EPOCHS = ", NUM_EPOCHS)
    while episode < NUM_EPOCHS:
        # trajectory = []
        distance = 0
        done = False
        uav.uav_reset(cell_objects)
        arrow_patch_list = reset_axes(ax_objects=ax_objects, cell_source=cell_source, cell_destination=cell_destination,
                                      arrow_patch_list=arrow_patch_list)
        # learner_feature_expectation = np.zeros(num_features, dtype=float)
        while distance < dist_infinite and not done:
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

            # features_current_state = phi_distance, phi_hop, phi_ues, phi_throughput, phi_interference

            # Choose an action based on epsilon-greedy
            # if random.random() < epsilon_grd:
            if random.random() < epsilon_decay:
                action = randint(0, len(action_list)-1)
            else:
                # Bring the model here for the greedy action
                action = get_greedy_action(sgd_models, features_current_state, std_scale)
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

            # learner_feature_expectation += get_feature_expectation(features_next_state, distance)

            # Calculate the reward
            immediate_reward = np.dot(weights, features_next_state)

            # Update the Next Q value and Calculate the td target
            q_value_next = sgd_predictor(sgd_models, features_next_state, std_scale)
            if new_cell == cell_destination:  # This is the termination point
                done = True
                q_td_target = immediate_reward
            else:
                q_td_target = immediate_reward + (gamma_discount * np.max(q_value_next))

            # Update the estimator(model)
            sgd_models, std_scale = update_sgd_models(sgd_models, features_current_state, action, q_td_target,
                                                      std_scale)

            arrow_patch_list = update_axes(ax_objects, prev_cell, cell_source, cell_destination, new_cell,
                                           action_power, cell_objects[new_cell].get_location(),
                                           action_movement, cell_objects[current_cell].get_location(), arrow_patch_list)
            # trajectory.append((features_current_state, (interference, sinr, throughput, interference_ues), action,
            #                    features_next_state, (interference_next, sinr_next, throughput_next,
            #                                          interference_ues_next),
            #                    immediate_reward, deepcopy(learner_feature_expectation)))
            prev_cell = new_cell
            distance += 1

        # if epsilon_decay > 0.001 and episode > num_required_replays:
        #     epsilon_decay -= (2 / NUM_EPOCHS)

        if epsilon_decay > 0.65 and episode > num_required_replays:
            epsilon_decay -= (1 / NUM_EPOCHS)

        # trajectory.append(learner_feature_expectation)
        # trajectories.append(trajectory)
        episode += 1
        if episode % 200 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            timer_end = time.perf_counter()
            print(" ......... EPISODE = ", episode, "......... Current Time = ", current_time,
                  " ..... ELAPSED TIME = ", round(timer_end - timer_start, 2), " Seconds, ",
                  round((timer_end - timer_start)/60, 2), " mins, ",
                  round((timer_end - timer_start)/3600, 2), " hour")

    if Config_Flags.get("PLOT_RESULTS"):
        plot_reward_irl_sgd(trajectories, learner_index)

    if Config_Flags.get("SAVE_IRL_DATA"):
        trajectories.append(sgd_models)
        trajectories.append(learner_index)
        learner_irl_file_name_np = 'Feature_%d_learner_%d_index_EPOCHS_%d' % (num_features, learner_index, NUM_EPOCHS)
        np.savez(InverseRLPath + learner_irl_file_name_np, trajectories=trajectories)

    if Config_Flags.get('SAVE_MODEL_IRL_SGD'):
        file_sgd_models_save = SGDModelPath + 'SGD_Feature_%d_learner_%d_index_EPOCHS_%d' % (num_features,
                                                                                             learner_index, NUM_EPOCHS)
        pickle.dump(sgd_models, open(file_sgd_models_save, 'wb'))

    return sgd_models
