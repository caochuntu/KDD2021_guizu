import re
import math
import numpy as np
from db_wrapper import get_node_id, get_node_gps_point

# Euclidean distance between two points
def euclidean_dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def linestring_to_point_array(linestring):
    # Linestring is in format: 'LINESTRING(lon1 lat1, lon2 lat2, ... , )'
    # First slice all unnecessary things off the string
    linestring = linestring[11:-1]
    # split the string into points
    points = linestring.split(',')
    # split to a tuple of long and lat
    points = [tuple(map(float, p.split())) for p in points]
    points = tuple(points)  # Order is important, so make it tuple
    return points
    
# Distance of point to linesegment
# u == vector from endpoints[0] to endpoints[1]
# v == vector from endpoints[0] to point
def point_to_lineseg_dist(endpoints, point):
    projection = get_projection(endpoints, point)
    return euclidean_dist(projection, point)

def get_projection(endpoints, point):
    endpoints = np.array(endpoints)
    p = np.array(point)
    u = endpoints[1] - endpoints[0]
    v = p - endpoints[0]
    # Magnitude of projection of v to u in terms of the magnitude of u
    projection_magnitude = np.dot(u,v) / np.dot(u,u)
    # If magnitude of projection is less than 0, it means that
    # the projection of point to line lies outside the linesegment
    # and the distance of point to linesegment is the distance from
    # point to endpoints[0]
    if projection_magnitude < 0:
        return endpoints[0]
    # Same as above
    elif projection_magnitude > 1:
        return endpoints[1]
    # If projection in [0,1], distance from point to line is the
    # distance from point to its orthogonal projection on the line
    projection = endpoints[0] + projection_magnitude*u
    return projection

def get_node_gps_points(matches):
    node_gps = []
    for i, match in enumerate(matches):
        if match['way_osm_id'] is None:
            node_ids.append(None)
            continue
        # Don't query the same point twice
        if i == 0 or match['way_osm_id'] != matches[i-1]['way_osm_id'] or match['index_in_way'] != matches[i-1]['index_in_way']:
            start_node = get_node_gps_point(match['way_osm_id'], match['index_in_way'])
            end_node = get_node_gps_point(match['way_osm_id'], match['index_in_way'] + 1)
        if match['direction'] == -1:
            node_gps.append((end_node, start_node))
        else:
            node_gps.append((start_node, end_node))
    return node_gps

def get_node_ids(matches):
    node_ids = []
    for i, match in enumerate(matches):
        if match['way_osm_id'] is None:
            node_ids.append(None)
            continue
        # Don't query the same point twice
        if i == 0 or match['way_osm_id'] != matches[i-1]['way_osm_id'] or match['index_in_way'] != matches[i-1]['index_in_way']:
            start_node = get_node_id(match['way_osm_id'], match['index_in_way'])
            start_node = re.findall(r'\d+', str(start_node))[0]
            end_node = get_node_id(match['way_osm_id'], match['index_in_way']+1)
            end_node = re.findall(r'\d+', str(end_node))[0]
        if match['direction'] == -1:
            node_ids.append((end_node, start_node))
        else:
            node_ids.append((start_node, end_node))
    return node_ids
 
def write_to_file(node_ids, filename):
    with open(filename, 'w') as f:
        f.write('Segment start id, Segment end id\n')
        for node in node_ids:
            if node is None:
                f.write('NA\n')
            else:
                f.write(node[0] + ', ' + node[1] + '\n')

# Calculates the direction we're traversing the segment based on previous segment
# If the segments are not the same, the start point of the segment is that point which
# is in the endpoints of the previous segment. If the segments are not connected, we 
# don't know the direction so return None. If the segment is the same as the previous segment,
# return previous segment's direction.
def calculate_direction(previous_segment, segment):
    if segment['endpoints'] == previous_segment['endpoints']:
        return previous_segment['direction']
    elif segment['endpoints'][0] in previous_segment['endpoints']:
        return 1
    elif segment['endpoints'][1] in previous_segment['endpoints']:
        return -1
    else:
        return None

def reverse_coordinate(coordinate):
    lon = coordinate[0]
    lat = coordinate[1]
    return (lat, lon)

