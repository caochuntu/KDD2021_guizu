import math
from geopy.distance import great_circle
import networkx as nx
from utils import reverse_coordinate
from db_wrapper import routingdistance
import sys
import time

BETA = 10

def heuristic(u, v):
    return 0

def routing_distance(g, segment1, segment2):
    orig_node = segment1['closest-node']
    dest_node = segment2['closest-node']
    if orig_node == dest_node:
        if segment1['furthest-node'] == segment2['furthest-node']:
            return great_circle(reverse_coordinate(segment1['closest-point']), reverse_coordinate(segment2['closest-point'])).meters
        else:
            return great_circle(reverse_coordinate(segment1['closest-point']), reverse_coordinate(segment1['closest-node-coordinate'])).meters + great_circle(reverse_coordinate(segment2['closest-point']), reverse_coordinate(segment2['closest-node-coordinate'])).meters
    if nx.has_path(g, orig_node, dest_node):
        route = nx.astar_path(g, orig_node, dest_node, heuristic, 'weight')
    else:
        return sys.maxint
    route_length_m = sum(g[u][v]['weight'] for u, v in zip(route[:-1], route[1:]))
    if segment1['furthest-node'] == route[1]:
        route_length_m = route_length_m - great_circle(reverse_coordinate(segment1['closest-point']), reverse_coordinate(segment1['closest-node-coordinate'])).meters
    else:
        route_length_m = route_length_m + great_circle(reverse_coordinate(segment1['closest-point']), reverse_coordinate(segment1['closest-node-coordinate'])).meters
    if segment2['furthest-node'] == route[-2]:
        route_length_m = route_length_m - great_circle(reverse_coordinate(segment2['closest-point']), reverse_coordinate(segment2['closest-node-coordinate'])).meters
    else:
        route_length_m = route_length_m + great_circle(reverse_coordinate(segment2['closest-point']), reverse_coordinate(segment2['closest-node-coordinate'])).meters
    return route_length_m

def transition_probability(g, segment1, segment2):
    route_distance = routing_distance(g, segment1, segment2)
    great_circle_distance = great_circle(reverse_coordinate(segment1['original-coordinate']), reverse_coordinate(segment2['original-coordinate'])).meters
    distance = abs(route_distance - great_circle_distance)
    if distance >= 2000:
        return -1.7976931348623157e+308
    transition_probability = math.exp(-1 * distance / BETA) * 1 / BETA
    log_transition = math.log(transition_probability)
    return log_transition


def calculate_transition_probability(g, segments1, segments2):
    transition_matrix = []
    for segment2 in segments2:
        transition_vector = []
        for segment1 in segments1:
            transition_vector.append(transition_probability(g, segment1, segment2))
        transition_matrix.append(transition_vector)
    return transition_matrix










