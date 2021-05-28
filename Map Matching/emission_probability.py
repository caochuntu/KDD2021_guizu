import numpy as np
import math
from db_wrapper import query_ways_within_radius, get_closest_point
from geopy.distance import great_circle
from utils import reverse_coordinate
import sys
import networkx as nx

sigma = 5

def calculate_emission_probability(g, lon, lat, radius):
    segments = query_ways_within_radius(g, lat, lon, radius)
    if segments == None:
        return None;
    coordinate = (lon, lat)
    for segment in segments:
        segment['closest-point'] = get_closest_point(lat, lon, segment)
        distance1 = great_circle(reverse_coordinate(segment['points'][0]), reverse_coordinate(segment['closest-point'])).meters
        distance2 = great_circle(reverse_coordinate(segment['points'][1]), reverse_coordinate(segment['closest-point'])).meters
        if distance1 < distance2:
            segment['closest-node'] = segment['node_ids'][0]
            segment['furthest-node'] = segment['node_ids'][1]
            segment['closest-node-coordinate'] = segment['points'][0]
        else:
            segment['closest-node'] = segment['node_ids'][1]
            segment['furthest-node'] = segment['node_ids'][0]
            segment['closest-node-coordinate'] = segment['points'][1]
        dist = great_circle(reverse_coordinate(coordinate), reverse_coordinate(segment['closest-point'])).meters
        if (dist > radius):
        	segment['emission-probability'] = -1.7976931348623157e+308
        else:
            emission_probability = (1/(math.sqrt(2*math.pi)*sigma))*math.exp(-0.5*(dist/sigma)**2)
            segment['emission-probability'] = math.log(emission_probability)
        segment['original-coordinate'] = coordinate
        segment['connected-to'] = [segment['points']]
        segment['matched-points'] = [(segment['original-coordinate'], segment['closest-point'])]
    return segments

def find_possible_segments(g, lon, lat, radius, states):
    segments = calculate_emission_probability(g, lon, lat,radius)
    if segments == None:
        return None, None
    probabilities = []
    temp_segments = []
    for segment in segments:
        if segment['emission-probability'] != -1.7976931348623157e+308:
            temp_segments.append(segment)
            probabilities.append(segment['emission-probability'])
    if len(temp_segments) == 0:
        return None, None
    if len(temp_segments) > states:
        combined = zip(temp_segments, probabilities)
        combined.sort(key=lambda el: -el[1])
        segments = [x[0] for x in combined]
        probabilities = [x[1] for x in combined]
        return segments[:states], probabilities[:states]
    return temp_segments, probabilities









