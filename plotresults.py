"""
#################################
# PLOT Module for demonstrating the results
#################################
"""

#########################################################
# import libraries
import pickle
import numpy as np
from config import Config_IRL
from config import Config_Path
import matplotlib.pyplot as plt
from config import Config_Flags


#########################################################
# General Parameters
num_features = Config_IRL.get('NUM_FEATURES')
ResultPathPDF = Config_Path.get('ResultPathPDF')
ResultPathFIG = Config_Path.get('ResultPathFIG')
#########################################################
# Function definition


def plot_reward_irl_sgd(trajectories, learner_index):
    # Trajectories:
    #           *: All Trajectories
    #                   1) Current Feature State
    #                   2) Current Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   3) Action
    #                   4) Next Feature State
    #                   5) Next Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   6) Immediate Reward (reward_location = 5 in the array)
    #                   7) Learner Feature Expectation (Immediate)
    #                   8) Learner Feature Expectation (Final)
    #           *: Final SGD Models for all Q-Action estimator
    #           *: Learner Index
    #
    #           INFO: Len(Trajectories) = NUM_EPOCHS(NUM EPISODES) + 1(SGD Models) + 1(Learner Index)
    #           INFO: Len(Trajectory)   = NUM Distance(max = dist_limit: 8) + 1 (learner_feature_expectation)
    #           INFO: Len(Each Step)    = NUM Elements(7)
    num_epochs = len(trajectories[0:-2])
    accumulative_reward = np.zeros(num_epochs, dtype=float)
    episode = 0
    reward_location = 5  # The reward location in each step of the trajectory array.

    for trajectory in trajectories[0:-2]:
        for step in trajectory[0:-1]:
            accumulative_reward[episode] += step[reward_location]
        episode += 1
    np.savetxt('SGD_accumulative_reward.csv',accumulative_reward)
    fig_reward = plt.figure(figsize=(8, 8))
    ax_reward = fig_reward.add_subplot(111)
    ax_reward.set_xlabel("EPOCHS", size=12, fontweight='bold')
    ax_reward.set_ylabel("Accumulative Reward", size=12, fontweight='bold')
    ax_reward.plot(np.arange(0, num_epochs) + 1, accumulative_reward, color="blue", linestyle='--', marker='o',
                   markersize='5', label='Accumulative Reward _ EPOCHs)', linewidth=2)
    ax_reward.grid(True)
    ax_reward.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj = ResultPathFIG + 'SGD_accumulative_reward_training_Feature_%d_learner_%d_EPOCHS_%d.fig.pickle' % \
                   (num_features, learner_index, num_epochs)
    file_fig_pdf = ResultPathPDF + 'SGD_accumulative_reward_training_Feature_%d_learner_%d_EPOCHS_%d.pdf' % \
                   (num_features, learner_index, num_epochs)

    if Config_Flags.get("SAVE_PLOT_PDF"):
        fig_reward.savefig(file_fig_pdf, bbox_inches='tight')
    if Config_Flags.get("SAVE_PLOT_FIG"):
        pickle.dump(fig_reward, open(file_fig_obj, 'wb'))


def plot_reward_irl_dqn(trajectories, learner_index):
    # trajectory.append((features_current_state, (interference, sinr, throughput, interference_ues), action,
    #                    features_next_state, (interference_next, sinr_next, throughput_next,
    #                                          interference_ues_next),
    #                    immediate_reward, deepcopy(learner_feature_expectation)))
    # trajectory.append(learner_feature_expectation)
    # trajectories.append(model)
    # trajectories.append(learner_index)
    # Trajectories:
    #           *: All Trajectories
    #                   1) Current Feature State
    #                   2) Current Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   3) Action
    #                   4) Next Feature State
    #                   5) Next Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   6) Immediate Reward (reward_location = 5 in the array)
    #                   7) Learner Feature Expectation (Immediate)
    #                   8) Learner Feature Expectation (Final)
    #           *: Final SGD Models for all Q-Action estimator
    #           *: Learner Index
    #
    #           INFO: Len(Trajectories) = NUM_EPOCHS(NUM EPISODES) + 1(SGD Models) + 1(Learner Index)
    #           INFO: Len(Trajectory)   = NUM Distance(max = dist_limit: 8) + 1 (learner_feature_expectation)
    #           INFO: Len(Each Step)    = NUM Elements(8)
    num_epochs = len(trajectories[0:-2])
    accumulative_reward = np.zeros(num_epochs, dtype=float)
    episode = 0
    reward_location = 5  # The reward location in each step of the trajectory array.

    for trajectory in trajectories[0:-2]:
        for step in trajectory[0:-1]:
            accumulative_reward[episode] += step[reward_location]
        episode += 1
    
    np.savetxt('DQN_accumulative_reward.csv',accumulative_reward)

    fig_reward = plt.figure(figsize=(8, 8))
    ax_reward = fig_reward.add_subplot(111)
    ax_reward.set_xlabel("EPOCHS", size=14, fontweight='bold')
    ax_reward.set_ylabel("Accumulative Reward", size=14, fontweight='bold')
    ax_reward.plot(np.arange(0, num_epochs) + 1, accumulative_reward, color="blue", linestyle='--', marker='o',
                   markersize='5', label='Accumulative Reward - EPOCHs)', linewidth=2)
    ax_reward.grid(True)
    ax_reward.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj = ResultPathFIG + 'DQN_accumulative_reward_training_Feature_%d_learner_%d_EPOCHS_%d.fig.pickle' % \
                   (num_features, learner_index, num_epochs)
    file_fig_pdf = ResultPathPDF + 'DQN_accumulative_reward_training_Feature_%d_learner_%d_EPOCHS_%d.pdf' % \
                   (num_features, learner_index, num_epochs)

    if Config_Flags.get("SAVE_PLOT_PDF"):
        fig_reward.savefig(file_fig_pdf, bbox_inches='tight')
    if Config_Flags.get("SAVE_PLOT_FIG"):
        pickle.dump(fig_reward, open(file_fig_obj, 'wb'))


def plot_training_trajectories(trajectories_sgd_run, trajectories_dqn_run, cell_objects):
    # trajectory.append((features_current_state, (interference, sinr, throughput, interference_ues), action,
    #                    features_next_state, (interference_next, sinr_next, throughput_next,
    #                                          interference_ues_next),
    #                    immediate_reward, deepcopy(learner_feature_expectation)))
    # trajectory.append(learner_feature_expectation)
    # trajectories.append(model)
    # trajectories.append(learner_index)
    # Trajectories:
    #           *: All Trajectories
    #                   1) Current Feature State
    #                   2) Current Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   3) Action
    #                   4) Next Feature State
    #                   5) Next Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   6) Immediate Reward (reward_location = 5 in the array)
    #                   7) New cell ID
    #
    #           INFO: Len(Trajectories) = NUM_EPOCHS(NUM EPISODES)
    #           INFO: Len(Trajectory)   = NUM Distance(max = dist_limit: 8)
    #           INFO: Len(Each Step)    = NUM Elements(7)
    num_epochs = len(trajectories_sgd_run[0])
    num_runs = len(trajectories_sgd_run)
    throughput_sgd = np.zeros((num_runs, num_epochs), dtype=float)
    throughput_dqn = np.zeros((num_runs, num_epochs), dtype=float)
    interference_ue_sgd = np.zeros((num_runs, num_epochs), dtype=float)
    interference_ue_dqn = np.zeros((num_runs, num_epochs), dtype=float)
    distance_destination_sgd = np.zeros((num_runs, num_epochs), dtype=int)
    distance_destination_dqn = np.zeros((num_runs, num_epochs), dtype=int)
    next_spec_index = 4
    throughput_index = 2
    interference_ue_index = 3
    new_cell_index = 6

    run = 0
    for trajectories_sgd, trajectories_dqn in zip(trajectories_sgd_run, trajectories_dqn_run):
        episode = 0
        for trajectory_sgd, trajectory_dqn in zip(trajectories_sgd, trajectories_dqn):
            distance_destination_sgd[run, episode] = cell_objects[trajectory_sgd[-1][new_cell_index]].get_distance()
            distance_destination_dqn[run, episode] = cell_objects[trajectory_dqn[-1][new_cell_index]].get_distance()
            for step_sgd, step_dqn in zip(trajectory_sgd, trajectory_dqn):
                throughput_sgd[run, episode] += step_sgd[next_spec_index][throughput_index]
                throughput_dqn[run, episode] += step_dqn[next_spec_index][throughput_index]
                interference_ue_sgd[run, episode] += step_sgd[next_spec_index][interference_ue_index]
                interference_ue_dqn[run, episode] += step_dqn[next_spec_index][interference_ue_index]
            episode += 1
        run += 1

    fig_train_throughput = plt.figure(figsize=(8, 8))
    ax_train_throughput = fig_train_throughput.add_subplot(111)
    ax_train_throughput.set_xlabel("EPOCHS", size=14, fontweight='bold')
    ax_train_throughput.set_ylabel("Average throughput UAV up link (Mbps)", size=14, fontweight='bold')
    ax_train_throughput.plot(100*np.arange(0, num_epochs) + 1, np.mean(throughput_sgd, axis=0), color="blue",
                             linestyle='--', marker='o', markersize='5', label='(Q-Learning)', linewidth=2)
    ax_train_throughput.plot(100*np.arange(0, num_epochs) + 1, np.mean(throughput_dqn, axis=0), color="red",
                             linestyle='--', marker='x', markersize='5', label='(DQN)', linewidth=2)
    ax_train_throughput.grid(True)
    ax_train_throughput.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj_throughput = ResultPathFIG + 'Throughput_learning_epochs_%d.fig.pickle' % \
                                num_epochs
    file_fig_pdf_throughput = ResultPathPDF + 'Throughput_learning_epochs_%d.pdf' % \
                   num_epochs

    fig_train_interference = plt.figure(figsize=(8, 8))
    ax_train_interference = fig_train_interference.add_subplot(111)
    ax_train_interference.set_xlabel("EPOCHS", size=14, fontweight='bold')
    ax_train_interference.set_ylabel("Summation of interference on UEs", size=14, fontweight='bold')
    ax_train_interference.plot(100*np.arange(0, num_epochs) + 1, np.mean(interference_ue_sgd, axis=0), color="blue",
                               linestyle='--', marker='o', markersize='5', label='(Q-Learning)', linewidth=2)
    ax_train_interference.plot(100*np.arange(0, num_epochs) + 1, np.mean(interference_ue_dqn, axis=0), color="red",
                               linestyle='--', marker='x', markersize='5', label='(DQN)', linewidth=2)
    ax_train_interference.grid(True)
    ax_train_interference.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj_interference = ResultPathFIG + 'Interference_learning_epochs_%d.fig.pickle' % \
                              num_epochs
    file_fig_pdf_interference = ResultPathPDF + 'Interference_learning_epochs_%d.pdf' % \
                              num_epochs

    fig_train_distance = plt.figure(figsize=(8, 8))
    ax_train_distance = fig_train_distance.add_subplot(111)
    ax_train_distance.set_xlabel("EPOCHS", size=14, fontweight='bold')
    ax_train_distance.set_ylabel("Average distance to the destination", size=14, fontweight='bold')
    ax_train_distance.plot(100*np.arange(0, num_epochs) + 1, np.mean(distance_destination_sgd, axis=0), color="blue",
                           linestyle='--', marker='o', markersize='5', label='(Q-Learning)', linewidth=2)
    ax_train_distance.plot(100*np.arange(0, num_epochs) + 1, np.mean(distance_destination_dqn, axis=0), color="red",
                           linestyle='--', marker='x', markersize='5', label='(DQN)', linewidth=2)
    ax_train_distance.grid(True)
    ax_train_distance.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj_distance = ResultPathFIG + 'Distance_learning_epochs_%d.fig.pickle' % \
                                num_epochs
    file_fig_pdf_distance = ResultPathPDF + 'Distance_learning_epochs_%d.pdf' % \
                                num_epochs

    if Config_Flags.get("SAVE_PLOT_PDF"):
        fig_train_throughput.savefig(file_fig_pdf_throughput, bbox_inches='tight')
        fig_train_interference.savefig(file_fig_pdf_interference, bbox_inches='tight')
        fig_train_distance.savefig(file_fig_pdf_distance, bbox_inches='tight')
    if Config_Flags.get("SAVE_PLOT_FIG"):
        pickle.dump(fig_train_throughput, open(file_fig_obj_throughput, 'wb'))
        pickle.dump(fig_train_interference, open(file_fig_obj_interference, 'wb'))
        pickle.dump(fig_train_distance, open(file_fig_obj_distance, 'wb'))


def plot_sample_trajectories(trajectories_sgd, trajectories_dqn, trajectories_bc, trajectories_shortest,
                             trajectories_random, cell_objects):
    # trajectory.append((features_current_state, (interference, sinr, throughput, interference_ues), action,
    #                    features_next_state, (interference_next, sinr_next, throughput_next,
    #                                          interference_ues_next),
    #                    immediate_reward, deepcopy(learner_feature_expectation)))
    # trajectory.append(learner_feature_expectation)
    # trajectories.append(model)
    # trajectories.append(learner_index)
    # Trajectories:
    #           *: All Trajectories
    #                   1) Current Feature State
    #                   2) Current Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   3) Action
    #                   4) Next Feature State
    #                   5) Next Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   6) Immediate Reward (reward_location = 5 in the array)
    #                   7) New cell ID
    num_steps_sgd = len(trajectories_sgd[0])
    num_steps_dqn = len(trajectories_dqn[0])
    num_steps_bc = len(trajectories_bc[0])
    num_steps_short = len(trajectories_shortest[0])
    num_steps_rand = len(trajectories_random[0])

    throughput_sgd = np.zeros(num_steps_sgd, dtype=float)
    interference_sgd = np.zeros(num_steps_sgd, dtype=float)
    distance_sgd = np.zeros(num_steps_sgd, dtype=int)

    throughput_dqn = np.zeros(num_steps_dqn, dtype=float)
    interference_dqn = np.zeros(num_steps_dqn, dtype=float)
    distance_dqn = np.zeros(num_steps_dqn, dtype=int)

    throughput_bc = np.zeros(num_steps_bc, dtype=float)
    interference_bc = np.zeros(num_steps_bc, dtype=float)
    distance_bc = np.zeros(num_steps_bc, dtype=int)

    throughput_short = np.zeros(num_steps_short, dtype=float)
    interference_short = np.zeros(num_steps_short, dtype=float)
    distance_short = np.zeros(num_steps_short, dtype=int)

    throughput_rand = np.zeros(num_steps_rand, dtype=float)
    interference_rand = np.zeros(num_steps_rand, dtype=float)
    distance_rand = np.zeros(num_steps_rand, dtype=int)

    next_spec_index = 4
    throughput_index = 2
    interference_ue_index = 3
    new_cell_index = 6

    for step, trajectory in enumerate(trajectories_sgd[0]):
        throughput_sgd[step] = trajectory[next_spec_index][throughput_index]
        interference_sgd[step] = trajectory[next_spec_index][interference_ue_index]
        distance_sgd[step] = cell_objects[trajectory[new_cell_index]].get_distance()

    for step, trajectory in enumerate(trajectories_dqn[0]):
        throughput_dqn[step] = trajectory[next_spec_index][throughput_index]
        interference_dqn[step] = trajectory[next_spec_index][interference_ue_index]
        distance_dqn[step] = cell_objects[trajectory[new_cell_index]].get_distance()

    for step, trajectory in enumerate(trajectories_bc[0]):
        throughput_bc[step] = trajectory[next_spec_index][throughput_index]
        interference_bc[step] = trajectory[next_spec_index][interference_ue_index]
        distance_bc[step] = cell_objects[trajectory[new_cell_index]].get_distance()

    for step, trajectory in enumerate(trajectories_shortest[0]):
        throughput_short[step] = trajectory[next_spec_index][throughput_index]
        interference_short[step] = trajectory[next_spec_index][interference_ue_index]
        distance_short[step] = cell_objects[trajectory[new_cell_index]].get_distance()

    for step, trajectory in enumerate(trajectories_random[0]):
        throughput_rand[step] = trajectory[next_spec_index][throughput_index]
        interference_rand[step] = trajectory[next_spec_index][interference_ue_index]
        distance_rand[step] = cell_objects[trajectory[new_cell_index]].get_distance()

    fig_sample_throughput = plt.figure(figsize=(8, 8))
    ax_sample_throughput = fig_sample_throughput.add_subplot(111)
    labels = ['Q-Learning', 'DQN', 'BC', 'Shortest', 'Random']
    x_pos = np.arange(len(labels))
    means = [np.mean(throughput_sgd), np.mean(throughput_dqn), np.mean(throughput_bc),
             np.mean(throughput_short), np.mean(throughput_rand)]
    errors = [np.std(throughput_sgd), np.std(throughput_dqn), np.std(throughput_bc),
              np.std(throughput_short), np.std(throughput_rand)]
    ax_sample_throughput.bar(x_pos, means, width=0.7, yerr=errors, align='center', alpha=0.5, ecolor='black',
                             capsize=10, color=['b', 'r', 'g', 'k', 'y', 'm', 'c'])
    ax_sample_throughput.set_xticks(x_pos)
    ax_sample_throughput.set_xticklabels(labels, size=14, fontweight='bold')
    ax_sample_throughput.set_ylabel('Average UAV throughput up link (Mbps)', size=14, fontweight='bold')
    ax_sample_throughput.set_title('Throughput mean and std for different approaches', size=14, fontweight='bold')
    ax_sample_throughput.yaxis.grid(True)

    file_fig_obj_throughput = ResultPathFIG + 'Throughput_sample_steps_%d.fig.pickle' % \
                              num_steps_sgd
    file_fig_pdf_throughput = ResultPathPDF + 'Throughput_sample_steps_%d.pdf' % \
                              num_steps_sgd

    fig_sample_interference = plt.figure(figsize=(8, 8))
    ax_sample_interference = fig_sample_interference.add_subplot(111)
    means_interference = [np.mean(interference_sgd), np.mean(interference_dqn), np.mean(interference_bc),
                          np.mean(interference_short), np.mean(interference_rand)]
    errors_interference = [np.std(interference_sgd), np.std(interference_dqn), np.std(interference_bc),
                            np.std(interference_short), np.std(interference_rand)]
    ax_sample_interference.bar(x_pos, means_interference, width=0.7, yerr=errors_interference, align='center',
                               alpha=0.5, ecolor='black', capsize=10, color=['b', 'r', 'g', 'k', 'y', 'm', 'c'])
    ax_sample_interference.set_xticks(x_pos)
    ax_sample_interference.set_xticklabels(labels, size=14, fontweight='bold')
    ax_sample_interference.set_ylabel('Average UAV interference on UEs DL', size=14, fontweight='bold')
    ax_sample_interference.set_title('Interference mean and std for different approaches', size=14, fontweight='bold')
    ax_sample_interference.yaxis.grid(True)

    file_fig_obj_interference = ResultPathFIG + 'Interference_sample_steps_%d.fig.pickle' % \
                              num_steps_sgd
    file_fig_pdf_interference = ResultPathPDF + 'Interference_sample_steps_%d.pdf' % \
                              num_steps_sgd

    fig_sample_distance = plt.figure(figsize=(8, 8))
    ax_sample_distance = fig_sample_distance.add_subplot(111)
    ax_sample_distance.set_xlabel("Steps", size=14, fontweight='bold')
    ax_sample_distance.set_ylabel("Minimum distance to the destination", size=14, fontweight='bold')
    ax_sample_distance.plot(np.arange(0, num_steps_sgd) + 1.15, distance_sgd, color="blue",
                             linestyle='solid', marker='o', markersize='5', label='Q-Learning', linewidth=1.5)
    ax_sample_distance.plot(np.arange(0, num_steps_dqn) + 0.85, distance_dqn, color="red",
                            linestyle='solid', marker='x', markersize='5', label='DQN', linewidth=1.5)
    ax_sample_distance.plot(np.arange(0, num_steps_bc) + 1, distance_bc, color="green",
                            linestyle='solid', marker='D', markersize='5', label='BC', linewidth=1.5)
    ax_sample_distance.plot(np.arange(0, num_steps_short) + 0.93, distance_short, color="k",
                            linestyle='solid', marker='v', markersize='5', label='Shortest path', linewidth=1.5)
    ax_sample_distance.plot(np.arange(0, num_steps_rand) + 1.07, distance_rand, color="y",
                            linestyle='solid', marker='s', markersize='5', label='Random', linewidth=1.5)
    ax_sample_distance.grid(True)
    ax_sample_distance.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj_distance = ResultPathFIG + 'Distance_sample_steps_%d.fig.pickle' % \
                                num_steps_sgd
    file_fig_pdf_distance = ResultPathPDF + 'Distance_sample_steps_%d.pdf' % \
                                num_steps_sgd

    if Config_Flags.get("SAVE_PLOT_PDF"):
        fig_sample_throughput.savefig(file_fig_pdf_throughput, bbox_inches='tight')
        fig_sample_interference.savefig(file_fig_pdf_interference, bbox_inches='tight')
        fig_sample_distance.savefig(file_fig_pdf_distance, bbox_inches='tight')
    if Config_Flags.get("SAVE_PLOT_FIG"):
        pickle.dump(fig_sample_throughput, open(file_fig_obj_throughput, 'wb'))
        pickle.dump(fig_sample_interference, open(file_fig_obj_interference, 'wb'))
        pickle.dump(fig_sample_distance, open(file_fig_obj_distance, 'wb'))


def plot_error_trajectories(trajectories_sgd, trajectories_dqn, trajectories_bc, cell_objects):
    num_steps_sgd = len(trajectories_sgd[0])
    num_steps_dqn = len(trajectories_dqn[0])
    num_steps_bc = len(trajectories_bc[0])

    throughput_sgd = np.zeros(num_steps_sgd, dtype=float)
    interference_sgd = np.zeros(num_steps_sgd, dtype=float)
    distance_sgd = np.zeros(num_steps_sgd, dtype=int)

    throughput_dqn = np.zeros(num_steps_dqn, dtype=float)
    interference_dqn = np.zeros(num_steps_dqn, dtype=float)
    distance_dqn = np.zeros(num_steps_dqn, dtype=int)

    throughput_bc = np.zeros(num_steps_bc, dtype=float)
    interference_bc = np.zeros(num_steps_bc, dtype=float)
    distance_bc = np.zeros(num_steps_bc, dtype=int)

    next_spec_index = 4
    throughput_index = 2
    interference_ue_index = 3
    new_cell_index = 6

    for step, trajectory in enumerate(trajectories_sgd[0]):
        throughput_sgd[step] = trajectory[next_spec_index][throughput_index]
        interference_sgd[step] = trajectory[next_spec_index][interference_ue_index]
        distance_sgd[step] = cell_objects[trajectory[new_cell_index]].get_distance()

    for step, trajectory in enumerate(trajectories_dqn[0]):
        throughput_dqn[step] = trajectory[next_spec_index][throughput_index]
        interference_dqn[step] = trajectory[next_spec_index][interference_ue_index]
        distance_dqn[step] = cell_objects[trajectory[new_cell_index]].get_distance()

    for step, trajectory in enumerate(trajectories_bc[0]):
        throughput_bc[step] = trajectory[next_spec_index][throughput_index]
        interference_bc[step] = trajectory[next_spec_index][interference_ue_index]
        distance_bc[step] = cell_objects[trajectory[new_cell_index]].get_distance()

    fig_error_throughput = plt.figure(figsize=(8, 8))
    ax_error_throughput = fig_error_throughput.add_subplot(111)
    labels = ['Q-Learning', 'DQN', 'BC']
    x_pos = np.arange(len(labels))
    means = [np.mean(throughput_sgd), np.mean(throughput_dqn), np.mean(throughput_bc)]
    errors = [np.std(throughput_sgd), np.std(throughput_dqn), np.std(throughput_bc)]
    ax_error_throughput.bar(x_pos, means, width=0.7, yerr=errors, align='center', alpha=0.5, ecolor='black',
                             capsize=10, color=['b', 'r', 'g', 'k', 'y', 'm', 'c'])
    ax_error_throughput.set_xticks(x_pos)
    ax_error_throughput.set_xticklabels(labels, size=14, fontweight='bold')
    ax_error_throughput.set_ylabel('Average UAV throughput - up link (Mbps)', size=14, fontweight='bold')
    ax_error_throughput.set_title('Throughput mean and std with environmental error', size=14, fontweight='bold')
    ax_error_throughput.yaxis.grid(True)

    file_fig_obj_throughput = ResultPathFIG + 'Throughput_error_steps_%d.fig.pickle' % \
                              num_steps_sgd
    file_fig_pdf_throughput = ResultPathPDF + 'Throughput_error_steps_%d.pdf' % \
                              num_steps_sgd

    fig_error_interference = plt.figure(figsize=(8, 8))
    ax_error_interference = fig_error_interference.add_subplot(111)
    means_interference = [np.mean(interference_sgd), np.mean(interference_dqn), np.mean(interference_bc)]
    errors_interference = [np.std(interference_sgd), np.std(interference_dqn), np.std(interference_bc)]
    ax_error_interference.bar(x_pos, means_interference, width=0.7, yerr=errors_interference, align='center',
                               alpha=0.5, ecolor='black', capsize=10, color=['b', 'r', 'g', 'k', 'y', 'm', 'c'])
    ax_error_interference.set_xticks(x_pos)
    ax_error_interference.set_xticklabels(labels, size=14, fontweight='bold')
    ax_error_interference.set_ylabel('Average UAV interference on UEs DL', size=14, fontweight='bold')
    ax_error_interference.set_title('Interference mean and std with environmental error', size=14, fontweight='bold')
    ax_error_interference.yaxis.grid(True)

    file_fig_obj_interference = ResultPathFIG + 'Interference_error_steps_%d.fig.pickle' % \
                                num_steps_sgd
    file_fig_pdf_interference = ResultPathPDF + 'Interference_error_steps_%d.pdf' % \
                                num_steps_sgd

    fig_error_distance = plt.figure(figsize=(8, 8))
    ax_error_distance = fig_error_distance.add_subplot(111)
    ax_error_distance.set_xlabel("Steps", size=14, fontweight='bold')
    ax_error_distance.set_ylabel("Minimum distance to the destination", size=14, fontweight='bold')
    ax_error_distance.plot(np.arange(0, num_steps_sgd) + 1.15, distance_sgd, color="blue",
                            linestyle='solid', marker='o', markersize='5', label='Q-Learning', linewidth=1.5)
    ax_error_distance.plot(np.arange(0, num_steps_dqn) + 0.85, distance_dqn, color="red",
                            linestyle='solid', marker='x', markersize='5', label='DQN', linewidth=1.5)
    ax_error_distance.plot(np.arange(0, num_steps_bc) + 1, distance_bc, color="green",
                            linestyle='solid', marker='D', markersize='5', label='BC', linewidth=1.5)
    ax_error_distance.grid(True)
    ax_error_distance.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj_distance = ResultPathFIG + 'Distance_error_steps_%d.fig.pickle' % \
                            num_steps_sgd
    file_fig_pdf_distance = ResultPathPDF + 'Distance_error_steps_%d.pdf' % \
                            num_steps_sgd
    if Config_Flags.get("SAVE_PLOT_PDF"):
        fig_error_throughput.savefig(file_fig_pdf_throughput, bbox_inches='tight')
        fig_error_interference.savefig(file_fig_pdf_interference, bbox_inches='tight')
        fig_error_distance.savefig(file_fig_pdf_distance, bbox_inches='tight')
    if Config_Flags.get("SAVE_PLOT_FIG"):
        pickle.dump(fig_error_throughput, open(file_fig_obj_throughput, 'wb'))
        pickle.dump(fig_error_interference, open(file_fig_obj_interference, 'wb'))
        pickle.dump(fig_error_distance, open(file_fig_obj_distance, 'wb'))
