import psycopg2
import re
import networkx as nx

DBNAME = 'minimiseddb'
USERNAME = 'jodiakyulas' 
LINE_TABLE = 'planet_osm_line'
LATLONG_DBNAME = 'osm_latlong'
ROUTEDBNAME = 'routing'

def connect(dbname):
    try:
        conn = psycopg2.connect("dbname='"+dbname+"' user='"+USERNAME+"' host='localhost'")
    except:
        print 'Unable to connect to database ' + dbname
        raise
    return conn.cursor()

# Query the database for ways that have nodes within 'radius' meters from the point defined
# by 'lat' and 'lon'
# Returns:
# 1) the point defined by 'lat' and 'lon' in mercator projection
# 2) An array containing dictionaries of all the ways that are within the radius.
#    The dictionary contains the osm_id of the way and a tuple of node-tuples in mercator projection.

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

def get_closest_point(lat, lon, segment):
    cur = connect(DBNAME)
    pointstr = "'POINT({0} {1})'".format(lon, lat)
    pointgenstr = 'ST_PointFromText({0}, {1})'.format(pointstr, 4326) 
    linestr = "'LINESTRING({0} {1},{2} {3})'".format(segment['points'][0][0], segment['points'][0][1], segment['points'][1][0], segment['points'][1][1])
    linegenstr = "ST_GeomFromText({0}, {1})".format(linestr, 4326)
    qstring = "SELECT ST_AsText(ST_ClosestPoint({0}, {1}));".format(linegenstr, pointgenstr)
    cur.execute(qstring)
    closest_point = cur.fetchone()[0]
    closest_point = re.findall(r"[-+]?\d*\.\d+|\d+", closest_point)
    closest_point = [float(d) for d in closest_point]
    closest_point = (closest_point[0], closest_point[1])
    return closest_point

def get_node_id(way_id, index):
    cur = connect(DBNAME)
    qstring = 'SELECT nodes[{0}] FROM planet_osm_ways WHERE id = {1}'.format(index+1, way_id)
    cur.execute(qstring)
    rows = cur.fetchall()
    if not len(rows):
        print way_id, index
    return rows[0] if len(rows) else None


def query_ways_within_radius(g, lat, lon, radius):
    cur = connect(DBNAME)
    # PostGIS format of a point. Long/Lat as this stage:
    pointstr = "'POINT({0} {1})'".format(lon, lat) 
    # PostGIS function to generate a point from text:
    pointgenstr = 'ST_PointFromText({0}, {1})'.format(pointstr, 4326) 
    # PostGIS function to transform point from lat/long to mercator:
    point_in_merc_str = 'ST_Transform({0}, {1})'.format(pointgenstr, 900913) 
    # Build query string from the pieces:
    qstring = "SELECT ST_AsText({1}), osm_id, ST_AsText(ST_Transform(way, {3})), oneway FROM {0} WHERE ST_DWithin(ST_Transform(way, {3}), {1}, {2});".format(LINE_TABLE, point_in_merc_str, radius, 900913)
    cur.execute(qstring)
    rows = cur.fetchall()
    # First cell of each row is the point in mercator projection as text.
    # Making the PostGIS database do the lat/long -> mercator conversion.
    # The point is in form 'POINT(long, lat)' so extract the floating point coordinates
    # with regex
    if not rows:
        return None
    point_in_merc = re.findall(r"[-+]?\d*\.\d+|\d+", rows[0][0])
    point_in_merc = [float(d) for d in point_in_merc]
    segments = []
    node_ids = []
    for row in rows:
        # second element of each row is the osm_id of the way
        osm_id = row[1]
        if osm_id < 0:
            continue
        # third element of each row is the linestring of the way as a string.
        # call linestring_to_point_array to convert the string into an array of points
        waygenstr = "ST_GeomFromText('{0}', {1})".format(row[2], 900913)
        qstring = "SELECT ST_AsText(ST_Transform({0}, {1}));".format(waygenstr, 4326)
        cur.execute(qstring)
        points = cur.fetchone()[0]
        point_array = linestring_to_point_array(points)
        for i, point in enumerate(point_array):
            if i != 0:
                previous_point = point_array[i - 1]
                current_point = point_array[i]
                previous_node_id = get_node_id(osm_id, i - 1)[0]
                current_node_id = get_node_id(osm_id, i)[0]
                previous_point = g.node[previous_node_id]['coordinate']
                current_point = g.node[current_node_id]['coordinate']
                temp_tuple = (previous_point, current_point)
                temp_nodes_tuple = (previous_node_id, current_node_id)
                if temp_tuple not in segments and temp_nodes_tuple not in node_ids:
                    segment = {}
                    segment['points'] = temp_tuple
                    segment['node_ids'] = temp_nodes_tuple
                    segments.append(segment)
                    node_ids.append(temp_nodes_tuple)
    return segments

def routingdistance(segment1, segment2):
    cur = connect(ROUTEDBNAME)
    way1str = "'POINT({0} {1})'".format(segment1['closest-point'][0], segment1['closest-point'][1]) 
    way2str = "'POINT({0} {1})'".format(segment2['closest-point'][0], segment2['closest-point'][1])

    way1_qstring = "SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> ST_GeometryFromText({0},4326) LIMIT 1;".format(way1str)
    way2_qstring = "SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> ST_GeometryFromText({0},4326) LIMIT 1;".format(way2str)

    cur.execute(way1_qstring)
    way1_id = cur.fetchone()[0]
    cur.execute(way2_qstring)
    way2_id = cur.fetchone()[0]

    if way1_id == way2_id:
        return 0

    qstring = "SELECT cost FROM pgr_aStar('SELECT gid AS id,source,target,length_m AS cost, reverse_cost, x1, y1, x2, y2 FROM ways', {0}, {1});".format(way1_id, way2_id)

    cur.execute(qstring)
    results = cur.fetchall()
    sum = 0;
    for result in results:
        sum += result[0]

    return sum



def get_node_gps_point(way_id, index):
    cur = connect(LATLONG_DBNAME)
    qstring = 'SELECT ST_AsText(way) FROM planet_osm_line WHERE osm_id = {0}'.format(way_id)
    cur.execute(qstring)
    rows = cur.fetchall()
    if not len(rows) or not len(rows[0]):
        return (None, None)
    points = utils.linestring_to_point_array(rows[0][0])
    return points[index] if len(points) else None


