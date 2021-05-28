from viterbi import run_viterbi
import numpy as np
import networkx as nx
import csv
import psycopg2
from geopy.distance import great_circle
from utils import reverse_coordinate
import osmgraph
import geog
import time
import sys
import math

radius = 50
states = 10
sigma = 5

dataset = "urecaTestData.csv"
output_matched_points = "result_quadtree_points.csv"
output_matched_segments = "result_quadtree_segments.csv"

sys.path.append('Pyqtree')
from pyqtree import *

print 'loading graph'
g = osmgraph.parse_file('test.osm')
for n1, n2 in g.edges():
	c1, c2 = osmgraph.tools.coordinates(g, (n1, n2))
	g[n1][n2]['weight'] = geog.distance(c1, c2)

def get_line_equation(x1, y1, x2, y2):
	try:
		m = (y1 - y2) / (x1 - x2)
	except:
		return None, None
	if m == 0:
		return None, None
	c = (y1 - m * x1)
	return -m, -c

def get_nearest_road_segments(currentlon, currentlat, radius):
	x1, y1, x2, y2 = get_bounding_box_for_radius(currentlon, currentlat, radius)
	BBOX_temp = (x1, y1, x2, y2)
	segments = []
	results = index.intersect_edges(BBOX_temp)
	for result in results:
		start = result.get_starting_node()
		end = result.get_ending_node()
		previous_point = g.node[start]['coordinate']
		current_point = g.node[end]['coordinate']
		temp_tuple = (previous_point, current_point)
		if temp_tuple not in segments:
			segment = {}
			segment['points'] = temp_tuple
			segment['node_ids'] = (start, end)
			segments.append(segment)
	return segments

def calculate_emission_probability(lon, lat, radius):
    segments = get_nearest_road_segments(lon, lat, radius)
    if len(segments) == 0:
    	return None
    coordinate = (lon, lat)
    for segment in segments:
        segment['closest-point'] = get_closest_point(segment['points'][0][0], segment['points'][0][1], segment['points'][1][0], segment['points'][1][1], lon, lat)
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
            if emission_probability == 0:
            	segment['emission-probability'] = -1.7976931348623157e+308
            else:
            	segment['emission-probability'] = math.log(emission_probability)
        segment['original-coordinate'] = coordinate
        segment['connected-to'] = [segment['points']]
        segment['matched-points'] = [(segment['original-coordinate'], segment['closest-point'])]
    return segments

def find_possible_segments(lon, lat, radius, states):
    segments = calculate_emission_probability(lon, lat, radius)
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


def get_closest_point(lon1, lat1, lon2, lat2, lon_test, lat_test):
	try:
		m = float(lat1 - lat2) / (lon1 - lon2)
	except:
		if lat1 < lat2:
			min_y = lat1
			max_y = lat2
		else:
			min_y = lat2
			max_y = lat1
		if (lat_test < min_y or lat_test > max_y):
			distance1 = great_circle((lat1, lon1), (lat_test, lon1)).meters
			distance2 = great_circle((lat2, lon2), (lat_test, lon1)).meters
			if distance1 < distance2:
				return (lon1, lat1)
			else:
				return (lon2, lat2)
		return (lon1, lat_test)
	if m == 0:
		if lon1 < lon2:
			min_x = lon1
			max_x = lon2
		else:
			min_x = lon2
			max_x = lon1
		if (lon_test < min_x or lon_test > max_x):
			distance1 = great_circle((lat1, lon1), (lat1, lon_test)).meters
			distance2 = great_circle((lat2, lon2), (lat1, lon_test)).meters
			if distance1 < distance2:
				return (lon1, lat1)
			else:
				return (lon2, lat2)
		return (lon_test, lat1)
	axis_intersect = lat1 - m * lon1
	a = -1 * m
	b = 1
	c = -1 * axis_intersect
	x = (b * (b * lon_test - a * lat_test) - a * c)/(math.pow(a, 2) + math.pow(b, 2))
	y = (a * (-b * lon_test + a * lat_test) - b * c)/(math.pow(a, 2) + math.pow(b, 2))
	if lon1 < lon2:
		min_x = lon1
		max_x = lon2
	else:
		min_x = lon2
		max_x = lon1
	if lat1 < lat2:
		min_y = lat1
		max_y = lat2
	else:
		min_y = lat2
		max_y = lat1
	if (y < min_y or y > max_y or x < min_x or x > max_x):
		distance1 = great_circle((lat1, lon1), (y, x)).meters
		distance2 = great_circle((lat2, lon2), (y, x)).meters
		if distance1 < distance2:
			return (lon1, lat1)
		else:
			return (lon2, lat2)
	return (x, y)

def get_bounding_box_for_radius(lon, lat, radius):
	R = 6378137
	dn = radius + 50
	de = radius + 50
	dLat = abs(float(dn) / R)
	dLon = abs(de / (R * math.cos(math.pi * lat / 180)))
	temp_xmax = lon + dLon * 180/math.pi
	temp_xmin = lon - dLon * 180/math.pi
	temp_ymax = lat + dLat * 180/math.pi
	temp_ymin = lat - dLat * 180/math.pi
	return temp_xmin, temp_ymin, temp_xmax, temp_ymax

def replacetemp(Results1, Results2):
	for j in range(0, len(Results2)):
		index_of_replacement = temp_table[j]
		Results1[index_of_replacement] = Results2[j]

def exportData(Results1, Results2):
	with open(output_matched_points, 'a') as out_file:
		for Result in Results1:
			if len(Result) == 2:
				rowString = str(Result[0][0]) + "," + str(Result[0][1]) + "," + str(Result[1][0]) + "," + str(Result[1][1]) + "\n"
			else:
				rowString = str(Result[0][0]) + "," + str(Result[0][1]) + "\n"
			out_file.write(rowString)
	out_file.close()

	with open(output_matched_segments, 'a') as out_file:
		for Result in Results2:
			if len(Result) == 2:
				rowString = str(Result[0][0]) + "," + str(Result[0][1]) + "," + str(Result[1][0]) + "," + str(Result[1][1]) + "\n"
			else:
				rowString = "N/A" + "\n"
			out_file.write(rowString)
	out_file.close()

xmax = 104.0395
ymax = 1.4713
xmin = 103.5884
ymin = 1.2098

data = []
matched_points = []
matched_segments = []

BBOX = (xmin, ymin, xmax, ymax)
index = Index(BBOX, None, None, None, None, 10, 20)

for n1 in g.nodes():
	c1 = osmgraph.tools.coordinates(g, (n1,))
	lon = c1[0][0]
	lat = c1[0][1]
	if lon < xmin or lon > xmax or lat < ymin or lat > ymax:
		continue
	BBOX_temp = (lon, lat, lon, lat)
	index.insert(n1, BBOX_temp)

for n1, n2 in g.edges():
	c1 = osmgraph.tools.coordinates(g, (n1,))
	x1 = c1[0][0]
	y1 = c1[0][1]
	c2 = osmgraph.tools.coordinates(g, (n2,))
	x2 = c2[0][0]
	y2 = c2[0][1]
	if n1 < n2:
		edge = Edge(n1, n2)
	else:
		edge = Edge(n2, n1)
	a, b = get_line_equation(x1, y1, x2, y2)
	if x1 < x2:
		x_min = x1
		x_max = x2
	else:
		x_min = x2
		x_max = x1
	if y1 < y2:
		y_min = y1
		y_max = y2
	else:
		y_min = y2
		y_max = y1
	BBOX_temp = (x_min, y_min, x_max, y_max)
	if a is None:
		index.insert_straight_edge(BBOX_temp, edge)
	else:
		index.insert_diagonal_edge(BBOX_temp, edge, a, b)

print 'graph loaded'

start_time = time.time()

with open(output_matched_points, 'w') as temp_file:
	print 'created a new out_file'
temp_file.close()

with open(output_matched_segments, 'w') as temp_file:
	print 'created a new out_file'
temp_file.close()

with open(dataset) as in_file:
	readCSV = csv.reader(in_file, delimiter = ',')
	for row in readCSV:
		rowString = (int(row[0]), row[1], float(row[2]), float(row[3]))
		data.append(rowString)
in_file.close()

previousstudentID = 0
previousdate = ""
previouslat = 0
previouslon = 0
start = True
i = 0
temp_table = []

observations = []

for datum in data:

	currentstudentID = data[i][0]
	currentdate = data[i][1][0:11]
	currentlat = data[i][2]
	currentlon = data[i][3]

	i = i + 1

	if start == True:
		previousstudentID = currentstudentID
		previousdate = currentdate
		previouslat = currentlat
		previouslon = currentlon
		start = False
		segments, em = find_possible_segments(currentlon, currentlat, radius, states)
		if segments == None:
			matched_points.append(((currentlon, currentlat), ))
			matched_segments.append((("N/A", ), ))
			continue
		temp_table.append(len(matched_points))
		matched_points.append("matched")
		matched_segments.append("matched")
		observations.append((segments, em))

	elif previousstudentID != currentstudentID:
		previousstudentID = currentstudentID
		previousdate = currentdate
		previouslat = currentlat
		previouslon = currentlon
		if len(observations) != 0:
			results1, results2 = run_viterbi(g, observations)
			replacetemp(matched_points, results1)
			replacetemp(matched_segments, results2)
		exportData(matched_points, matched_segments)
		matched_points[:] = []
		matched_segments[:] = []
		observations[:] = []
		temp_table[:] = []
		segments, em = find_possible_segments(currentlon, currentlat, radius, states)
		if segments == None:
			matched_points.append(((currentlon, currentlat), ))
			matched_segments.append((("N/A", ), ))
			continue
		temp_table.append(len(matched_points))
		matched_points.append("matched")
		matched_segments.append("matched")
		observations.append((segments, em))

	elif previousdate != currentdate:
		previousstudentID = currentstudentID
		previousdate = currentdate
		previouslat = currentlat
		previouslon = currentlon
		if len(observations) != 0:
			results1, results2 = run_viterbi(g, observations)
			replacetemp(matched_points, results1)
			replacetemp(matched_segments, results2)
		exportData(matched_points, matched_segments)
		matched_points[:] = []
		matched_segments[:] = []
		temp_table[:] = []
		observations[:] = []
		segments, em = find_possible_segments(currentlon, currentlat, radius, states)
		if segments == None:
			matched_points.append(((currentlon, currentlat), ))
			matched_segments.append((("N/A", ), ))
			continue
		temp_table.append(len(matched_points))
		matched_points.append("matched")
		matched_segments.append("matched")
		observations.append((segments, em))

	else:
		previouslat = currentlat
		previouslon = currentlon
		segments, em = find_possible_segments(currentlon, currentlat, radius, states)
		if segments == None:
			matched_points.append(((currentlon, currentlat), ))
			matched_segments.append((("N/A", ), ))
			continue
		temp_table.append(len(matched_points))
		matched_points.append("matched")
		matched_segments.append("matched")
		observations.append((segments, em))

if len(observations) != 0:
	results1, results2 = run_viterbi(g, observations)
	replacetemp(matched_points, results1)
	replacetemp(matched_segments, results2)
exportData(matched_points, matched_segments)

print time.time() - start_time, "seconds"

