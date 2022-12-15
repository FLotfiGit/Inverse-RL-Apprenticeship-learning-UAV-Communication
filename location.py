"""
#################################
# Location and allocation functions and modules
#################################
"""

#########################################################
# import libraries
import numpy as np
import matplotlib.pyplot as plt
from config import Config_Flags
from config import Config_General
from utils import power_to_radius
#from hexalattice.hexalattice import *
from matplotlib.patches import RegularPolygon, Arrow


#########################################################
# General Parameters
first_arrow, arrow_patch = True, None
radius = Config_General.get('Radius')
cells = Config_General.get('NUM_CELLS')
num_ues = Config_General.get('NUM_UEs')
loc_delta = Config_General.get('Loc_delta')

#########################################################
# Function and definition


def plothexagon():
    fig_cells, ax_cells = plt.subplots(1, figsize=(12, 8))
    ax_cells.set_aspect('equal')
    coordinates = [[None, float] for _ in range(0, cells)]
    cell = 0
    for ind_x in range(0, np.int(np.sqrt(cells))):
        for ind_y in range(0, np.int(np.sqrt(cells))):
            coordinates[cell][:] = [ind_x, ind_y]
            cell += 1
    cell_ids = [np.int(np.sqrt(cells))*coord[0] + coord[1] for coord in coordinates]
    hcoord = [(3./2.) * radius * coord[0] for coord in coordinates]
    vcoord = [-(np.mod(coord[0], 2))*((np.sqrt(3.)/2.) * radius) + ((coord[1]) * (np.sqrt(3.)) * radius) for coord
              in coordinates]

    for x, y, cid in zip(hcoord, vcoord, cell_ids):
        hexagon = RegularPolygon((x, y), numVertices=6, radius=radius, edgecolor='k', facecolor='g', alpha=0.25,
                                 orientation=np.float(np.radians(30)), linewidth=1.5)
        ax_cells.add_patch(hexagon)
        ax_cells.text(x, y+((np.sqrt(3.)/8.) * radius), cid, ha='center', va='center', size=12)

    # hex_centers, _ = create_hex_grid(n=40,
    #                                  do_plot=True,
    #                                  rotate_deg=30.0,
    #                                  face_color=[0, 0.6, 0.4])
    ax_cells.scatter(hcoord, vcoord, color='b', alpha=0.8, marker='^', s=50)
    ax_cells.patches[0].set_color('r')
    ax_cells.patches[cells-1].set_color('r')

    circle = plt.Circle((0, 0), radius=radius*0.5*np.sqrt(3), color='b', alpha=0.3)
    ax_cells.add_artist(circle)
    ax_cells.set_xlim([min(hcoord) - 2 * radius, max(hcoord) + 2 * radius])
    ax_cells.set_ylim([min(vcoord) - 2 * radius, max(vcoord) + 2 * radius])
    ax_cells.set_xlabel("X - Location", size=14, fontweight='bold')
    ax_cells.set_ylabel("Y - Location", size=14, fontweight='bold')
    ax_cells.grid(True)
    # plt.show(block=False)
    return np.array(vcoord), np.array(hcoord), cell_ids, fig_cells, ax_cells, coordinates


def plotues(fig_cells, ax_cells, cell_ids, hcoord, vcoord):
    x_coord_ues, y_coord_ues = geo_data_75ues_25cells(hcoord, vcoord)
    ax_cells.scatter(x_coord_ues[:], y_coord_ues[:], color='m', edgecolors='none', marker='o')
    # ax_cells.scatter(x_coord_ues, y_coord_ues, color='m', alpha=0.01)
    if Config_Flags.get('Display_map'):
        plt.show(block=False)
    return fig_cells, ax_cells, x_coord_ues, y_coord_ues


def geo_data_75ues_25cells(hcoord, vcoord):
    x_coord_ues = np.zeros([num_ues], dtype=float)
    y_coord_ues = np.zeros([num_ues], dtype=float)

    x_coord_ues[0], y_coord_ues[0] = hcoord[1] - 3.0 * loc_delta, vcoord[1] + 0.0 * loc_delta
    x_coord_ues[1], y_coord_ues[1] = hcoord[1] + 0.0 * loc_delta, vcoord[1] + 3.0 * loc_delta

    x_coord_ues[2], y_coord_ues[2] = hcoord[2] - 2.0 * loc_delta, vcoord[2] - 2.0 * loc_delta
    x_coord_ues[3], y_coord_ues[3] = hcoord[2] - 0.0 * loc_delta, vcoord[2] + 2.0 * loc_delta

    x_coord_ues[4], y_coord_ues[4] = hcoord[3] - 1.0 * loc_delta, vcoord[3] - 3.0 * loc_delta
    x_coord_ues[5], y_coord_ues[5] = hcoord[3] + 1.0 * loc_delta, vcoord[3] + 3.0 * loc_delta

    x_coord_ues[6], y_coord_ues[6] = hcoord[4] - 0.0 * loc_delta, vcoord[4] + 3.0 * loc_delta
    x_coord_ues[7], y_coord_ues[7] = hcoord[4] + 0.5 * loc_delta, vcoord[4] - 3.0 * loc_delta
    x_coord_ues[8], y_coord_ues[8] = hcoord[4] - 3.0 * loc_delta, vcoord[4] + 1.0 * loc_delta

    x_coord_ues[9], y_coord_ues[9] = hcoord[5] - 2.0 * loc_delta, vcoord[5] - 2.0 * loc_delta
    x_coord_ues[10], y_coord_ues[10] = hcoord[5] + 1.0 * loc_delta, vcoord[5] - 3.0 * loc_delta
    x_coord_ues[11], y_coord_ues[11] = hcoord[5] + 2.0 * loc_delta, vcoord[5] + 2.0 * loc_delta

    x_coord_ues[12], y_coord_ues[12] = hcoord[6] - 3.0 * loc_delta, vcoord[6] - 3.0 * loc_delta
    x_coord_ues[13], y_coord_ues[13] = hcoord[6] + 2.0 * loc_delta, vcoord[6] + 2.0 * loc_delta
    x_coord_ues[14], y_coord_ues[14] = hcoord[6] - 1.0 * loc_delta, vcoord[6] + 2.0 * loc_delta

    x_coord_ues[15], y_coord_ues[15] = hcoord[7] + 3.0 * loc_delta, vcoord[7] + 2.5 * loc_delta
    x_coord_ues[16], y_coord_ues[16] = hcoord[7] - 3.0 * loc_delta, vcoord[7] + 1.0 * loc_delta
    x_coord_ues[17], y_coord_ues[17] = hcoord[7] - 2.0 * loc_delta, vcoord[7] - 3.0 * loc_delta

    x_coord_ues[18], y_coord_ues[18] = hcoord[8] + 0.0 * loc_delta, vcoord[8] + 3.0 * loc_delta
    x_coord_ues[19], y_coord_ues[19] = hcoord[8] - 2.0 * loc_delta, vcoord[8] + 1.0 * loc_delta
    x_coord_ues[20], y_coord_ues[20] = hcoord[8] - 1.0 * loc_delta, vcoord[8] - 2.0 * loc_delta

    x_coord_ues[21], y_coord_ues[21] = hcoord[9] - 2.5 * loc_delta, vcoord[9] + 2.5 * loc_delta
    x_coord_ues[22], y_coord_ues[22] = hcoord[9] + 3.0 * loc_delta, vcoord[9] + 2.0 * loc_delta

    x_coord_ues[23], y_coord_ues[23] = hcoord[10] - 2.0 * loc_delta, vcoord[10] + 2.0 * loc_delta
    x_coord_ues[24], y_coord_ues[24] = hcoord[10] + 3.0 * loc_delta, vcoord[10] + 1.0 * loc_delta
    x_coord_ues[25], y_coord_ues[25] = hcoord[10] - 2.5 * loc_delta, vcoord[10] - 2.5 * loc_delta
    x_coord_ues[26], y_coord_ues[26] = hcoord[10] + 0.0 * loc_delta, vcoord[10] - 3.0 * loc_delta
    x_coord_ues[27], y_coord_ues[27] = hcoord[10] + 1.0 * loc_delta, vcoord[10] - 2.7 * loc_delta

    x_coord_ues[28], y_coord_ues[28] = hcoord[11] - 0.0 * loc_delta, vcoord[11] - 2.5 * loc_delta
    x_coord_ues[29], y_coord_ues[29] = hcoord[11] + 2.5 * loc_delta, vcoord[11] - 1.7 * loc_delta
    x_coord_ues[30], y_coord_ues[30] = hcoord[11] + 2.5 * loc_delta, vcoord[11] + 1.8 * loc_delta
    x_coord_ues[31], y_coord_ues[31] = hcoord[11] + 1.0 * loc_delta, vcoord[11] + 2.9 * loc_delta

    x_coord_ues[32], y_coord_ues[32] = hcoord[12] + 1.0 * loc_delta, vcoord[12] - 2.8 * loc_delta
    x_coord_ues[33], y_coord_ues[33] = hcoord[12] + 3.0 * loc_delta, vcoord[12] + 0.0 * loc_delta
    x_coord_ues[34], y_coord_ues[34] = hcoord[12] - 1.0 * loc_delta, vcoord[12] + 1.0 * loc_delta

    x_coord_ues[35], y_coord_ues[35] = hcoord[13] + 1.0 * loc_delta, vcoord[13] + 2.0 * loc_delta
    x_coord_ues[36], y_coord_ues[36] = hcoord[13] - 2.5 * loc_delta, vcoord[13] + 0.0 * loc_delta

    x_coord_ues[37], y_coord_ues[37] = hcoord[14] + 0.0 * loc_delta, vcoord[14] + 2.0 * loc_delta
    x_coord_ues[38], y_coord_ues[38] = hcoord[14] - 2.5 * loc_delta, vcoord[14] + 1.3 * loc_delta

    x_coord_ues[39], y_coord_ues[39] = hcoord[15] + 0.0 * loc_delta, vcoord[15] - 3.0 * loc_delta
    x_coord_ues[40], y_coord_ues[40] = hcoord[15] + 2.7 * loc_delta, vcoord[15] - 1.0 * loc_delta
    x_coord_ues[41], y_coord_ues[41] = hcoord[15] + 1.5 * loc_delta, vcoord[15] + 2.7 * loc_delta
    x_coord_ues[42], y_coord_ues[42] = hcoord[15] - 1.0 * loc_delta, vcoord[15] + 2.7 * loc_delta
    x_coord_ues[43], y_coord_ues[43] = hcoord[15] - 3.0 * loc_delta, vcoord[15] + 0.0 * loc_delta

    x_coord_ues[44], y_coord_ues[44] = hcoord[16] + 2.7 * loc_delta, vcoord[16] - 1.0 * loc_delta
    x_coord_ues[45], y_coord_ues[45] = hcoord[16] + 1.5 * loc_delta, vcoord[16] + 2.7 * loc_delta
    x_coord_ues[46], y_coord_ues[46] = hcoord[16] - 1.0 * loc_delta, vcoord[16] + 2.7 * loc_delta
    x_coord_ues[47], y_coord_ues[47] = hcoord[16] - 3.0 * loc_delta, vcoord[16] + 0.0 * loc_delta

    x_coord_ues[48], y_coord_ues[48] = hcoord[17] + 0.0 * loc_delta, vcoord[17] - 3.0 * loc_delta
    x_coord_ues[49], y_coord_ues[49] = hcoord[17] + 0.0 * loc_delta, vcoord[17] + 2.7 * loc_delta
    x_coord_ues[50], y_coord_ues[50] = hcoord[17] - 3.0 * loc_delta, vcoord[17] - 0.8 * loc_delta
    x_coord_ues[51], y_coord_ues[51] = hcoord[17] + 1.0 * loc_delta, vcoord[17] - 1.0 * loc_delta
    x_coord_ues[52], y_coord_ues[52] = hcoord[17] - 3.0 * loc_delta, vcoord[17] + 0.0 * loc_delta
    x_coord_ues[53], y_coord_ues[53] = hcoord[17] + 2.5 * loc_delta, vcoord[17] + 1.5 * loc_delta

    x_coord_ues[54], y_coord_ues[54] = hcoord[18] - 0.0 * loc_delta, vcoord[18] - 2.0 * loc_delta
    x_coord_ues[55], y_coord_ues[55] = hcoord[18] + 2.5 * loc_delta, vcoord[18] - 1.5 * loc_delta

    x_coord_ues[56], y_coord_ues[56] = hcoord[19] - 2.5 * loc_delta, vcoord[19] + 0.5 * loc_delta
    x_coord_ues[57], y_coord_ues[57] = hcoord[19] + 2.5 * loc_delta, vcoord[19] - 0.5 * loc_delta

    x_coord_ues[58], y_coord_ues[58] = hcoord[20] + 2.5 * loc_delta, vcoord[20] + 1.5 * loc_delta
    x_coord_ues[59], y_coord_ues[59] = hcoord[20] + 2.5 * loc_delta, vcoord[20] - 1.9 * loc_delta
    x_coord_ues[60], y_coord_ues[60] = hcoord[20] - 1.8 * loc_delta, vcoord[20] - 0.5 * loc_delta

    x_coord_ues[61], y_coord_ues[61] = hcoord[21] + 1.5 * loc_delta, vcoord[21] + 3.0 * loc_delta
    x_coord_ues[62], y_coord_ues[62] = hcoord[21] - 1.5 * loc_delta, vcoord[21] + 3.0 * loc_delta
    x_coord_ues[63], y_coord_ues[63] = hcoord[21] - 2.8 * loc_delta, vcoord[21] + 0.2 * loc_delta
    x_coord_ues[64], y_coord_ues[64] = hcoord[21] - 2.5 * loc_delta, vcoord[21] - 2.5 * loc_delta
    x_coord_ues[65], y_coord_ues[65] = hcoord[21] - 0.8 * loc_delta, vcoord[21] - 0.2 * loc_delta
    x_coord_ues[66], y_coord_ues[66] = hcoord[21] + 2.8 * loc_delta, vcoord[21] - 1.9 * loc_delta

    x_coord_ues[67], y_coord_ues[67] = hcoord[22] - 2.8 * loc_delta, vcoord[22] - 1.2 * loc_delta
    x_coord_ues[68], y_coord_ues[68] = hcoord[22] + 2.5 * loc_delta, vcoord[22] + 2.5 * loc_delta
    x_coord_ues[69], y_coord_ues[69] = hcoord[22] + 1.9 * loc_delta, vcoord[22] - 1.2 * loc_delta
    x_coord_ues[70], y_coord_ues[70] = hcoord[22] - 1.8 * loc_delta, vcoord[22] - 1.9 * loc_delta
    x_coord_ues[71], y_coord_ues[71] = hcoord[22] - 1.8 * loc_delta, vcoord[22] + 1.9 * loc_delta

    x_coord_ues[72], y_coord_ues[72] = hcoord[23] + 1.8 * loc_delta, vcoord[23] + 0.4 * loc_delta
    x_coord_ues[73], y_coord_ues[73] = hcoord[23] - 1.8 * loc_delta, vcoord[23] - 1.9 * loc_delta
    x_coord_ues[74], y_coord_ues[74] = hcoord[23] + 2.8 * loc_delta, vcoord[23] - 2.9 * loc_delta
    return x_coord_ues, y_coord_ues


def update_axes(ax_objects, prev_cell, cell_source, cell_destination, neighbor_rand, tx_power, center, action,
                arrow_center, arrow_list):
    global first_arrow, arrow_patch
    ax_objects.patches[prev_cell].set_color('g')
    ax_objects.patches[cell_source].set_color('r')
    ax_objects.patches[cell_destination].set_color('r')
    ax_objects.patches[neighbor_rand].set_color('b')
    tx_radius = power_to_radius(tx_power)
    # ***************************************
    # Keep single circle for demonstration
    # ax_objects.artists[0].set_center(center[0:2])
    # ax_objects.artists[0].set_radius(tx_radius)
    # ***************************************
    # Add multiple circles for demonstration
    circle = plt.Circle(center[0:2], radius=tx_radius, color='b', alpha=0.3)
    ax_objects.add_artist(circle)
    # ***************************************
    dx, dy = action_to_arrow(action)

    if Config_Flags.get('SingleArrow'):
        # ***************************************
        # Just the latest(recent) arrow or action
        arrow = Arrow(arrow_center[0], arrow_center[1], dx, dy, width=3, fc='k')
        if first_arrow:
            arrow_patch = ax_objects.add_patch(arrow)
            first_arrow = False
        else:
            arrow_patch.remove()
            arrow_patch = ax_objects.add_patch(arrow)
    else:
        # ***************************************
        # Multiple arrows for all taken actions
        arrow_item = ax_objects.arrow(arrow_center[0], arrow_center[1], dx, dy, head_width=1.05, head_length=1.1,
                                      fc='k', ec='k')
        arrow_list.append(arrow_item)
    return arrow_list


def reset_axes(ax_objects, cell_source, cell_destination, arrow_patch_list):
    ax_objects.patches[cell_source].set_color('r')
    ax_objects.patches[cell_destination].set_color('r')
    for cell in range(cell_source+1, cell_destination):
        ax_objects.patches[cell].set_color('g')
    #ax_objects.artists[0].set_center((0, 0))  #@fl
    #ax_objects.artists[0].set_radius(radius*0.5*np.sqrt(3))  #@fl
    for arrow_item in arrow_patch_list:
        arrow_item.remove()
    arrow_patch_list = []
    return arrow_patch_list


def action_to_arrow(action):
    a_len = radius * 0.5 * np.sqrt(3) - 1
    dx = 0
    dy = 0
    if action == 1:
        dx = 0
        dy = a_len
    elif action == 2:
        dx = a_len * 0.5 * np.sqrt(3)
        dy = a_len / 2
    elif action == 3:
        dx = a_len * 0.5 * np.sqrt(3)
        dy = -a_len / 2
    elif action == 4:
        dx = 0
        dy = -a_len
    elif action == 5:
        dx = -a_len * 0.5 * np.sqrt(3)
        dy = -a_len / 2
    elif action == 6:
        dx = -a_len * 0.5 * np.sqrt(3)
        dy = a_len / 2
    else:
        exit('Error: Not a defined action for the movement')
    return dx, dy
