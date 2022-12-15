"""
#################################
# Utility functions
#################################
"""

#########################################################
# import libraries
import numpy as np
from config import Config_Power
from config import Config_General
from config import Config_interference
from config import config_movement_step
from config import movement_actions_list
from scipy.spatial.distance import cdist
from config import Number_of_neighbor_UEs
from scipy.spatial.distance import euclidean

#########################################################
# General Parameters
radius = Config_General.get('Radius')
num_ues = Config_General.get("NUM_UEs")
num_cells = Config_General.get("NUM_CELLS")
tx_powers = Config_Power.get('UAV_Tr_power')
ue_tr_power = Config_Power.get("UE_Tr_power")
bandwidth = Config_interference.get('Bandwidth')
float_acc = Config_General.get('FLOAT_ACCURACY')
antenna_gain = Config_interference.get('AntennaGain')

#########################################################
# Class and Function definitions

# ******************************************************
# ******************* CELL CLASS ***********************
# ******************************************************


class Cell:

    def __init__(self, x_loc=None, y_loc=None, num_ues_cell=-1, unique_id=-1):
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.z_loc = 0.0
        self.num_ues_cell = num_ues_cell
        self.num_neighbor_ues = 0
        self.cell_id = unique_id
        self.location = [self.x_loc, self.y_loc, self.z_loc]
        self.ues_idx = None
        self.coordinate = None
        self.neighbors = None
        self.available_actions = None
        self.dist_destination = float('inf')

    def print_info(self):
        print("Cell ID = ", self.cell_id, "\n",
              "Location = ", self.location, "\n",
              "Num UEs = ", self.num_ues_cell, "\n",
              "UEs index = ", self.ues_idx, "\n",
              "Neighbors = ", self.neighbors, "\n",
              "Actions = ", self.available_actions, "\n")

    def set_location(self, loc):
        self.x_loc = loc[0]
        self.y_loc = loc[1]
        self.location = [self.x_loc, self.y_loc, self.z_loc]

    def set_num_ues(self, num_ues_cell):
        self.num_ues_cell = num_ues_cell

    def set_num_neighbor_ues(self, num_neighbor_ues):
        self.num_neighbor_ues = num_neighbor_ues

    def set_id(self, uid):
        self.cell_id = uid

    def set_ues_ids(self, ues_idx):
        self.ues_idx = ues_idx

    def set_coord(self, coord):
        self.coordinate = coord

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def set_available_actions(self, actions):
        self.available_actions = actions

    def set_distance(self, distance):
        self.dist_destination = distance

    def get_location(self):
        return self.location

    def get_num_ues(self):
        return self.num_ues_cell

    def get_num_neighbor_ues(self):
        return self.num_neighbor_ues

    def get_id(self):
        return self.cell_id

    def get_ues_idx(self):
        return self.ues_idx

    def get_coord(self):
        return self.coordinate

    def get_neighbor(self):
        return self.neighbors

    def get_actions(self):
        return self.available_actions

    def get_distance(self):
        return self.dist_destination

# ******************************************************
# ******************* UAV CLASS ***********************
# ******************************************************


class UAV:

    def __init__(self, x_loc=None, y_loc=None, z_loc=None, cell_id=-1, tr_power=0):
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.z_loc = z_loc
        self.cell_id = cell_id
        self.power = tr_power
        self.location = [self.x_loc, self.y_loc, self.z_loc]
        self.action_movement = 0
        self.interference = 0
        self.snr = 0
        self.sinr = 0
        self.throughput = 0
        self.throughput_snr = 0
        self.interference_over_ues = 0
        self.hop = 0

    def set_location(self, loc):
        self.x_loc = loc[0]
        self.y_loc = loc[1]
        self.location = [self.x_loc, self.y_loc, self.z_loc]

    def set_cell_id(self, cid):
        self.cell_id = cid

    def set_power(self, tr_power):
        self.power = tr_power

    def set_action_movement(self, action):
        self.action_movement = action

    def set_hop(self, hop):
        self.hop = hop

    def get_location(self):
        return self.location

    def get_cell_id(self):
        return self.cell_id

    def get_tr_power(self):
        return self.power

    def get_action_movement(self):
        return self.action_movement

    def get_hop(self):
        return self.hop

    def send_pkt(self):
        pass

    def calc_interference(self, cells_objects, ues_objects):
        current_cell = self.get_cell_id()
        neighbors = cells_objects[current_cell].get_neighbor()
        interference = 0
        for neighbor in neighbors:
            ues = cells_objects[neighbor].get_ues_idx()
            for ue in ues:
                csi = get_csi(ues_objects[ue].get_location(), cells_objects[current_cell].get_location())
                interference += (ues_objects[ue].get_power()) * ((abs(csi))**2)
                # print(interference)
        self.interference = interference
        return self.interference

    def calc_sinr(self, cell_objects):
        cell = self.get_cell_id()
        csi = get_csi(self.location, cell_objects[cell].get_location())
        csi_abs = (abs(csi))**2
        sinr = (self.get_tr_power()) * csi_abs / (1 + self.interference)
        snr = (self.get_tr_power()) * csi_abs
        self.sinr = sinr
        self.snr = snr
        return self.sinr, self.snr

    def calc_throughput(self):
        throughput = np.log2(1 + self.sinr)
        throughput_snr = np.log2(1 + self.snr)
        self.throughput = throughput
        self.throughput_snr = throughput_snr
        return self.throughput_snr
        # return self.throughput

    def calc_max_throughput(self, cell_objects):
        cell = self.get_cell_id()
        csi = get_csi(self.location, cell_objects[cell].get_location())
        csi_abs = (abs(csi))**2
        max_power = max(tx_powers)
        snr = max_power * csi_abs
        throughput_max = np.log2(1 + snr)
        return throughput_max

    def calc_interference_ues(self, cells_objects, ues_objects):
        current_cell = self.get_cell_id()
        neighbors = cells_objects[current_cell].get_neighbor()
        interference = 0
        for neighbor in neighbors:
            ues = cells_objects[neighbor].get_ues_idx()
            for ue in ues:
                csi = get_csi(self.location, ues_objects[ue].get_location())
                interference_ue = self.power * ((abs(csi))**2)
                # print(interference_ue)
                ues_objects[ue].set_interference(interference_ue)
                interference += interference_ue
        self.interference_over_ues = interference
        return self.interference_over_ues

    def get_interference(self):
        return self.interference

    def get_sinr(self):
        return self.sinr, self.snr

    def get_throughput(self):
        return self.throughput

    def uav_perform_task(self, cell_objects, ues_objects):
        interference = self.calc_interference(cell_objects, ues_objects)
        sinr, snr = self.calc_sinr(cell_objects)
        throughput = self.calc_throughput()
        interference_ues = self.calc_interference_ues(cell_objects, ues_objects)
        max_throughput = self.calc_max_throughput(cell_objects=cell_objects)
        return interference, sinr, throughput, interference_ues, max_throughput

    def uav_reset(self, cell_objects):
        self.set_cell_id(cid=0)
        self.set_location(loc=cell_objects[0].get_location())
        self.set_hop(hop=0)
        self.set_power(tr_power=0)

# ******************************************************
# ******************* UE CLASS ***********************
# ******************************************************


class UE:

    def __init__(self, x_loc=None, y_loc=None, ue_id=-1, cell_id=-1, tr_power=0):
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.z_loc = 0.0
        self.ue_id = ue_id
        self.cell_id = cell_id
        self.power = tr_power
        self.location = [self.x_loc, self.y_loc, self.z_loc]
        self.interference = 0

    def set_location(self, loc):
        self.x_loc = loc[0]
        self.y_loc = loc[1]
        self.location = [self.x_loc, self.y_loc, self.z_loc]

    def set_ue_id(self, ue_id):
        self.ue_id = ue_id

    def set_cell_id(self, cell_id):
        self.cell_id = cell_id

    def set_power(self, tr_power):
        self.power = tr_power

    def set_interference(self, interference):
        self.interference = interference

    def get_location(self):
        return self.location

    def get_ue_id(self):
        return self.ue_id

    def get_cell_id(self):
        return self.cell_id

    def get_power(self):
        return self.power

    def get_interference(self):
        return self.interference


def find_closest_cell(h_coord_cells, v_coord_cells, x_coord_ues, y_coord_ues):
    ue_cell_ids = np.zeros([num_ues], dtype=np.int16) - 1
    cell_coord_pairs = np.concatenate((h_coord_cells.reshape(-1, 1), v_coord_cells.reshape(-1, 1)), axis=1)
    for index in range(0, num_ues):
        dist = cdist(np.array([[x_coord_ues[index], y_coord_ues[index]]]), cell_coord_pairs, 'euclidean')
        min_index = np.argmin(dist)
        ue_cell_ids[index] = min_index
    return ue_cell_ids


def create_ues(x_coord_ues, y_coord_ues, ue_cell_ids):
    ues_objects = np.empty(num_ues, dtype=object)
    for ue in range(0, num_ues):
        ues_objects[ue] = UE(x_loc=x_coord_ues[ue], y_loc=y_coord_ues[ue])
        ues_objects[ue].set_ue_id(ue)
        ues_objects[ue].set_cell_id(ue_cell_ids[ue])
        ues_objects[ue].set_power(ue_tr_power)
    return ues_objects


def create_cells(h_coord_cells, v_coord_cells, cell_ids, ue_cell_ids, coordinates):
    cells_objects = np.empty(num_cells, dtype=object)
    counts = np.zeros(num_cells, dtype=np.int16)
    _, counts[0+1:num_cells-1] = np.unique(ue_cell_ids, return_counts=True)
    for cell in range(0, num_cells):
        cells_objects[cell] = Cell(h_coord_cells[cell], v_coord_cells[cell])
        cells_objects[cell].set_id(cell_ids[cell])
        cells_objects[cell].set_num_ues(counts[cell])
        cells_objects[cell].set_ues_ids(np.where(ue_cell_ids == cell)[0])
        cells_objects[cell].set_coord(coordinates[cell])

    for cell in range(0, num_cells):
        available_neighbor, available_action = find_neighbors(cells_objects[cell], cells_objects)
        cells_objects[cell].set_neighbors(available_neighbor)
        cells_objects[cell].set_available_actions(available_action)

    for cell in range(0, num_cells):
        num_neighbor_ues = 0
        for neighbor in cells_objects[cell].get_neighbor():
            num_neighbor_ues += cells_objects[neighbor].get_num_ues()
        cells_objects[cell].set_num_neighbor_ues(num_neighbor_ues)

    list_neighbor_ues_ordered = []
    for cell in range(0, num_cells):
        list_neighbor_ues_ordered.append(cells_objects[cell].get_num_neighbor_ues())
    # print("Sorted Summation Number of Neighbor UEs: ", sorted(list_neighbor_ues_ordered))
    Number_of_neighbor_UEs['Min'], Number_of_neighbor_UEs['Max'] = min(list_neighbor_ues_ordered),\
                                                                   max(list_neighbor_ues_ordered)

    cell = num_cells - 1
    dest_cell = cell_ids[-1]
    source_cell = cell_ids[0]
    while cell >= 0:
        if cells_objects[cell].get_id() == dest_cell:
            cells_objects[cell].set_distance(0)
        else:
            neighbors = cells_objects[cell].get_neighbor()
            distances = []
            for neighbor in neighbors:
                distances.append(cells_objects[neighbor].get_distance())
                cells_objects[cell].set_distance(int(min(distances)) + 1)
        # print("cell = ", cell, " distance = ", cells_objects[cell].get_distance())
        cell -= 1

    return cells_objects


def check_neighbor_availability(location, cells_objects):
    for cell in range(0, len(cells_objects)):
        if round(cells_objects[cell].get_location()[0], float_acc) == round(location[0], float_acc) and \
                round(cells_objects[cell].get_location()[1], float_acc) == round(location[1], float_acc):
            return True, cells_objects[cell].get_id()
    return False, None


def find_neighbors(cell_object, cell_objects):
    available_action = []
    available_neighbor = []
    x_cell = cell_object.get_location()[0]
    y_cell = cell_object.get_location()[1]
    for action in movement_actions_list:
        x_change, y_change = action_to_location(action)
        new_x = x_cell + x_change
        new_y = y_cell + y_change
        check_flag, neighbor = check_neighbor_availability([new_x, new_y], cell_objects)
        if check_flag:
            available_action.append(action)
            available_neighbor.append(neighbor)
    return available_neighbor, available_action


def action_to_location(action):
    x_change, y_change = None, None
    x_step, y_step = config_movement_step.get('x_step'), config_movement_step.get('y_step')
    if action == 1:
        x_change = 0
        y_change = y_step
    elif action == 2:
        x_change = x_step
        y_change = (1./2.) * y_step
    elif action == 3:
        x_change = x_step
        y_change = (-1./2.) * y_step
    elif action == 4:
        x_change = 0
        y_change = -y_step
    elif action == 5:
        x_change = -x_step
        y_change = (-1./2.) * y_step
    elif action == 6:
        x_change = -x_step
        y_change = (1./2.) * y_step
    else:
        exit('Error: Not a defined action for the movement')
    return x_change, y_change

####################################### @fl
def generate_rician(K,data_size):
    rician_mu  = np.sqrt(K/(K+1))
    rician_s   = np.sqrt(1/(2*(K+1)))
    rician_chn = rician_s*(np.random.standard_normal((data_size,)) + np.random.standard_normal((data_size,))*1j) + rician_mu
    return rician_chn


def get_csi(loc_source, loc_destination):
    distance = euclidean(loc_source, loc_destination)
    csi = antenna_gain * (1/distance**2) * (1 + 1j)
    #---------------------------------------- @fl
    gamma = 2  # free space 
    K = 1 # num RBs
    #K_rician = 1.59
    #csi_rnd = antenna_gain*(distance**(-gamma))*(np.randn(1 , K)+np.randn(1 , K)*1j)/(2**0.5) 
    #ch_rician = generate_rician(K_rician,K)
    #csi_rnd_rician = antenna_gain*(distance**(-gamma))*ch_rician
    #--------
    n_L = 0.1
    n_NL = 21
    c1 = 12.076 #4.879
    c2 = 0.1139 #0.429
    H = Config_General.get('Altitude')
    pL = 1/ (1 + c1 * np.exp(-c2*(180/np.pi * np.arctan(H/distance)-c1)))
    path_loss = (distance**(gamma)) * (pL * n_L  +  (1-pL) * n_NL  )
    csi_n =  antenna_gain /path_loss
    return csi_n  #csi 


def power_to_radius(power):
    min_power = min(tx_powers)
    max_power = max(tx_powers)
    diff = power - min_power
    radius_circle = (diff/(max_power-min_power)) * radius * np.sqrt(3) + 0.5 * radius * np.sqrt(3)
    return radius_circle


def multi_actions_to_action(action_movement, action_tx):
    return np.where(action_tx == np.array(tx_powers))[0] * len(movement_actions_list) + \
           np.where(action_movement == np.array(movement_actions_list))[0]


def action_to_multi_actions(action):
    action_movement = int(action % len(movement_actions_list))
    action_tx = int(action / len(movement_actions_list))
    return action_movement, action_tx
