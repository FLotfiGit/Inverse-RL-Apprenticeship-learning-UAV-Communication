"""
#################################
# Imitation Learning: Behavioral Cloning
#################################
"""

#########################################################
# import libraries
import time
import pickle
import numpy as np
from tqdm import tqdm
from random import seed
from sklearn import svm
from sklearn import tree
from datetime import datetime
from config import Config_IRL
from config import Config_Path
from config import Config_Power
from config import Config_Flags
from location import reset_axes
import matplotlib.pyplot as plt
from location import update_axes
from config import Config_General
from xgboost import XGBClassifier
from config import Config_requirement
from config import Number_of_neighbor_UEs
from utils import multi_actions_to_action
from utils import action_to_multi_actions
from sklearn.metrics import accuracy_score
from config import Config_BehavioralCloning
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

#########################################################
# General Parameters
num_cells = Config_General.get('NUM_CELLS')
BCModelPath = Config_Path.get('BCModelPath')
tx_powers = Config_Power.get('UAV_Tr_power')
num_features = Config_IRL.get('NUM_FEATURES')
ExpertPath_BC = Config_Path.get('ExpertPath_BC')
dist_limit = Config_requirement.get('dist_limit')
MAX_DISTANCE = Config_requirement.get('MAX_DISTANCE')
MIN_UE_NEIGHBORS = Config_requirement.get('MIN_UE_NEIGHBORS')
MAX_UE_NEIGHBORS = Config_requirement.get('MAX_UE_NEIGHBORS')
MIN_INTERFERENCE = Config_requirement.get('MIN_INTERFERENCE')
MAX_INTERFERENCE = Config_requirement.get('MAX_INTERFERENCE')
NUM_EPISODES = Config_BehavioralCloning.get('NUM_TRAJECTORIES_EXPERT')

seed(1369)
cell_source = 0
cell_destination = num_cells - 1
#########################################################
# Function definition


def behavioral_cloning(uav, ues_objects, ax_objects, cell_objects):
    print(" ****** Mode: Behavioral cloning for the drone ")
    trajectories = load_expert_trajectories(uav, ues_objects, ax_objects, cell_objects, load_data=True)
    models = train_model_behavioral(trajectories, load_model=True)
    imitation_behavioral_cloning(uav, ues_objects, ax_objects, cell_objects, models)


def load_expert_trajectories(uav, ues_objects, ax_objects, cell_objects, load_data=False):
    if not load_data:
        episode = 0
        prev_cell = 1
        trajectories = []
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
                current_state = uav.get_cell_id()
                current_features = get_features(state=cell, cell_objects=cell_objects, uav=uav, ues_objects=ues_objects)
                action = choose_action_expert(cell)
                trajectories.append((current_features, action))
                if Config_Flags.get('PRINT_INFO'):
                    print('Chosen Action: ', action)
                action_movement_index, action_tx_index = action_to_multi_actions(action)
                expert_action_mov = action_movement_index + 1
                expert_action_power = tx_powers[action_tx_index]

                # expert_action_mov = int(input("Please select the cell to move" + str(movement_actions_list) + ": "))

                avail_actions_mov = cell_objects[cell].get_actions()
                avail_neighbors = cell_objects[cell].get_neighbor()
                if np.any(expert_action_mov == np.array(avail_actions_mov)):
                    new_state = avail_neighbors[np.where(expert_action_mov == np.array(avail_actions_mov))[0][0]]
                else:
                    new_state = current_state

                new_cell = new_state
                uav.set_cell_id(cid=new_cell)
                uav.set_location(loc=cell_objects[new_cell].get_location())
                uav.set_hop(hop=uav.get_hop() + 1)

                suggested_power = min(tx_powers) + (Number_of_neighbor_UEs.get('Max') -
                                                    cell_objects[new_state].get_num_neighbor_ues()) / \
                                  (Number_of_neighbor_UEs.get('Max') - Number_of_neighbor_UEs.get('Min')) * \
                                  (max(tx_powers) - min(tx_powers))
                if Config_Flags.get('PRINT_INFO'):
                    print("\n********** INFO:\n Number of Neighbor UEs: ",
                          cell_objects[new_state].get_num_neighbor_ues(),
                          '\n', "Suggested Power: ", suggested_power)
                # expert_action_power = float(input("Please select the TX Power" + str(tx_powers) + ":"))
                expert_action = multi_actions_to_action(expert_action_mov, expert_action_power)
                if Config_Flags.get('PRINT_INFO'):
                    print("expert_action: ", expert_action)
                uav.set_power(tr_power=expert_action_power)

                interference_next, sinr_next, throughput_next, interference_ues_next, max_throughput_next, = \
                    uav.uav_perform_task(cell_objects, ues_objects)
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
                          "Interference on Neighbor UEs: ", interference_ues_next)
                # features = get_features(state=new_cell, cell_objects=cell_objects, uav=uav, ues_objects=ues_objects)

                arrow_patch_list = update_axes(ax_objects, prev_cell, cell_source, cell_destination, new_cell,
                                               expert_action_power, cell_objects[new_cell].get_location(),
                                               expert_action_mov, cell_objects[cell].get_location(), arrow_patch_list)
                if Config_Flags.get('Display_map'):
                    plt.pause(0.001)
                prev_cell = new_cell
                if new_cell == cell_destination:
                    done = True
                distance += 1

            episode += 1
            if episode % 200 == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                timer_end = time.perf_counter()
                print(" ......... EPISODE = ", episode, "......... Current Time = ", current_time,
                      " ..... ELAPSED TIME = ", round(timer_end - timer_start, 2), " Seconds, ",
                      round((timer_end - timer_start)/60, 2), " mins, ",
                      round((timer_end - timer_start)/3600, 2), " hour")

        if Config_Flags.get("SAVE_DATA_BC_EXPERT"):
            file_name = 'Expert_BehavioralCloning_Features_%d_trajectories_%d' % (num_features, NUM_EPISODES)
            np.savez(ExpertPath_BC + file_name, trajectories=trajectories)
        return trajectories
    else:
        file_name = 'Expert_BehavioralCloning_Features_%d_trajectories_%d.npz' % (num_features, NUM_EPISODES)
        trajectories = np.load(ExpertPath_BC + file_name, allow_pickle=True).get('trajectories')
        return trajectories


def get_features(state, cell_objects, uav, ues_objects):
    phi_distance = np.power((cell_objects[state].get_distance()) / MAX_DISTANCE, 2.)
    # phi_hop = 1 - np.power((uav.get_hop()) / dist_limit, 2.)
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


def choose_action_expert(cell):
    action = -1
    if cell == 0:
        action = 24
    elif cell == 1:
        action = 24
    elif cell == 2:
        action = np.random.choice([24, 19])
    elif cell == 3:
        action = 25
    elif cell == 9:
        action = 31
    elif cell == 14:
        action = 26
    elif cell == 19:
        action = 31
    elif cell == 8:
        action = np.random.choice([24, 19])
    elif cell == 13:
        action = 25

    if action == -1:
        exit(' ........... Exit: wrong action, wrong cell')
    return action


def train_model_behavioral(trajectories, load_model=False):
    # Useful links and helps:
    num_trajectories = len(trajectories)
    x_input = np.zeros((num_trajectories, num_features), dtype=float)
    y_input = np.zeros(num_trajectories, dtype=int)
    for index, trajectory in tqdm(enumerate(trajectories)):
        x_input[index, :] = trajectory[0]
        y_input[index] = trajectory[1]

    train_x, test_x, train_y, test_y = train_test_split(x_input, y_input, test_size=0.2)
    if not load_model:

        clf_sgd = SGDClassifier(loss="hinge", penalty="l2")
        clf_sgd.fit(train_x, train_y)

        y_predicted_sgd = clf_sgd.predict(test_x)
        print('Accuracy: {:.2f}'.format(accuracy_score(test_y, y_predicted_sgd)))

        clf_svm = svm.SVC(decision_function_shape='ovo')
        clf_svm.fit(train_x, train_y)
        y_predicted_svm = clf_svm.predict(test_x)
        print('Accuracy: {:.2f}'.format(accuracy_score(test_y, y_predicted_svm)))

        clf_tree = tree.DecisionTreeClassifier()
        clf_tree = clf_tree.fit(train_x, train_y)
        y_predicted_tree = clf_tree.predict(test_x)
        print('Accuracy: {:.2f}'.format(accuracy_score(test_y, y_predicted_tree)))

        clf_gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1,
                                                                      random_state=0).fit(train_x, train_y)
        clf_gradient_boosting_classifier.score(test_x, test_y)

        clf_xgb = XGBClassifier().fit(train_x, train_y)
        y_pred_xgb = clf_xgb.predict(test_x)
        # predictions = [round(value) for value in y_pred_xgb]
        accuracy_xgb = accuracy_score(test_y, y_pred_xgb)
        print("Accuracy: %.2f%%" % (accuracy_xgb * 100.0))

        if Config_Flags.get('SAVE_MODEL_BC'):
            file_bc_models_save = BCModelPath + 'BC_Models_Feature_%d_EPOCHS_%d' % (num_features, NUM_EPISODES)
            pickle.dump((clf_sgd, clf_svm, clf_tree, clf_gradient_boosting_classifier, clf_xgb),
                        open(file_bc_models_save, 'wb'))

        return clf_sgd, clf_svm, clf_tree, clf_gradient_boosting_classifier, clf_xgb
    else:
        file_bc_models_save = BCModelPath + 'BC_Models_Feature_%d_EPOCHS_%d' % (num_features, NUM_EPISODES)
        with open(file_bc_models_save, "rb") as file_obj:
            models = pickle.load(file_obj)
            clf_sgd, clf_svm, clf_tree, clf_gradient_boosting_classifier, clf_xgb = models[0], models[1], models[2],\
                                                                                    models[3], models[4]
            y_predicted_sgd = clf_sgd.predict(test_x)
            print('Accuracy SGD: {:.2f}'.format(accuracy_score(test_y, y_predicted_sgd)))
            y_predicted_svm = clf_svm.predict(test_x)
            print('Accuracy SVM: {:.2f}'.format(accuracy_score(test_y, y_predicted_svm)))
            y_predicted_tree = clf_tree.predict(test_x)
            print('Accuracy Decision Tree: {:.2f}'.format(accuracy_score(test_y, y_predicted_tree)))
            gradient_boosting_classifier_result = clf_gradient_boosting_classifier.score(test_x, test_y)
            print('Accuracy GBC: {:.2f}'.format(gradient_boosting_classifier_result))
            y_predicted_xgb = clf_xgb.predict(test_x)
            accuracy_xgb = accuracy_score(test_y, y_predicted_xgb)
            print("Accuracy XGB: %.2f%%" % (accuracy_xgb * 100.0))

        return models


def imitation_behavioral_cloning(uav, ues_objects, ax_objects, cell_objects, models):
    episode = 0
    prev_cell = 1
    trajectories = []
    arrow_patch_list = []
    timer_start = time.perf_counter()
    print("......... TOTAL EPOCHS = ", NUM_EPISODES)
    clf_sgd, clf_svm, clf_tree, clf_gradient_boosting_classifier, clf_xgb = models[0], models[1], models[2], \
                                                                            models[3], models[4]
    # fig = plt.figure(figsize=(10, 15))
    # tree.plot_tree(clf_tree, filled=True)
    # fig.savefig("file_fig_pdf.pdf", bbox_inches='tight')
    while episode < NUM_EPISODES:
        distance = 0
        done = False
        arrow_patch_list = reset_axes(ax_objects=ax_objects, cell_source=cell_source,
                                      cell_destination=cell_destination, arrow_patch_list=arrow_patch_list)
        uav.uav_reset(cell_objects)

        while distance < dist_limit and not done:
            cell = uav.get_cell_id()
            current_state = uav.get_cell_id()
            current_features = get_features(state=cell, cell_objects=cell_objects, uav=uav, ues_objects=ues_objects)
            action = clf_xgb.predict(np.array(current_features).reshape(1, -1))
            trajectories.append((current_features, action))

            action_movement_index, action_tx_index = action_to_multi_actions(action)
            expert_action_mov = action_movement_index + 1
            expert_action_power = tx_powers[action_tx_index]
            avail_actions_mov = cell_objects[cell].get_actions()
            avail_neighbors = cell_objects[cell].get_neighbor()
            if np.any(expert_action_mov == np.array(avail_actions_mov)):
                new_state = avail_neighbors[np.where(expert_action_mov == np.array(avail_actions_mov))[0][0]]
            else:
                new_state = current_state

            new_cell = new_state
            uav.set_cell_id(cid=new_cell)
            uav.set_location(loc=cell_objects[new_cell].get_location())
            uav.set_hop(hop=uav.get_hop() + 1)
            expert_action = multi_actions_to_action(expert_action_mov, expert_action_power)
            if Config_Flags.get('PRINT_INFO'):
                print("Chosen Action: ", expert_action, '\n',
                      "Chosen Mobility: ", expert_action_mov, '\n',
                      "Chosen Power: ", expert_action_power)

            uav.set_power(tr_power=expert_action_power)
            interference_next, sinr_next, throughput_next, interference_ues_next, max_throughput_next, = \
                uav.uav_perform_task(cell_objects, ues_objects)
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
                      "Interference on Neighbor UEs: ", interference_ues_next)

            arrow_patch_list = update_axes(ax_objects, prev_cell, cell_source, cell_destination, new_cell,
                                           expert_action_power, cell_objects[new_cell].get_location(),
                                           expert_action_mov, cell_objects[cell].get_location(), arrow_patch_list)
            if Config_Flags.get('Display_map'):
                plt.pause(0.001)
            prev_cell = new_cell
            if new_cell == cell_destination:
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

    return 0
