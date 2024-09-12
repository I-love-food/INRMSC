import sys
from test_fun import *

REGULAR = -1
MAXIMUM = 0
MINIMUM = 1
SADDLE = 2
MONKEY_SADDLE = 3 # to do

def greater(mesh, vertex1, vertex2):
    value1 = test_fun(mesh.points[vertex1])
    value2 = test_fun(mesh.points[vertex2])
    if value1 != value2:
        return value1 > value2
    return vertex1 > vertex2

def smaller(mesh, vertex1, vertex2):
    value1 = test_fun(mesh.points[vertex1])
    value2 = test_fun(mesh.points[vertex2])
    if value1 != value2:
        return value1 < value2
    return vertex1 < vertex2
    
def on_boundary(p, h, v):
    return abs(p[0] - h[0]) <= 1e-5 or abs(p[0] - h[1]) <= 1e-5 or abs(p[1] - v[0]) <= 1e-5 or abs(p[1] - v[1]) <= 1e-5

def get_vertex_neighbours(vertex_id, mesh):
    index_pointers, indices = mesh.vertex_neighbor_vertices
    result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
    return result_ids

def get_vertex_neighours_from(vertex_id, mesh):
    index_pointers, indices = mesh.vertex_neighbor_vertices
    result_ids = indices[index_pointers[vertex_id]:index_pointers[-1]]
    return result_ids

def trace_descending(vertex_id, mesh):
    current_vertex = vertex_id
    path = []
    while True:
        path.append(current_vertex)
        next_vertex = current_vertex
        neighbours = get_vertex_neighbours(current_vertex, mesh)
        for candidate in neighbours:
            if smaller(mesh, candidate, next_vertex):
                next_vertex = candidate
        if next_vertex != current_vertex:
            current_vertex = next_vertex
        else:
            break
    return path

def trace_ascending(vertex_id, mesh):
    current_vertex = vertex_id
    path = []
    while True:
        path.append(current_vertex)
        neighbours = get_vertex_neighbours(current_vertex, mesh)
        next_vertex = current_vertex
        for candidate in neighbours:
            if greater(mesh, candidate, next_vertex):
                next_vertex = candidate
        if next_vertex != current_vertex:
            current_vertex = next_vertex
        else:
            break
    return path

def get_connected_components(mesh, link):
    P = []
    def find(x):
        if P[x] != x:
            P[x] = find(P[x])
        return P[x]
    def merge(x, y):
        px = find(x)
        py = find(y)
        P[py] = px
    for i in range(len(link)):
        P.append(i)
    neighbours = []
    for i in link:
        neighbours.append(set(get_vertex_neighbours(i, mesh)))
    for i in range(len(link)):
        for j in range(i + 1, len(link)):
            if link[i] in neighbours[j]:
                merge(i, j)
    connected_component = {}
    for i in P:
        if connected_component.get(find(i)) == None:
            connected_component[find(i)] = []
        connected_component[find(i)].append(i)
    return list(connected_component.values())

def point_identification(mesh, vertex_id): # 假设函数一定是morse function
    upper_link = []
    lower_link = []
    neighbours = get_vertex_neighbours(vertex_id, mesh)
    for star_vertex in neighbours:
        if greater(mesh, star_vertex, vertex_id):
            upper_link.append(star_vertex)
        if smaller(mesh, star_vertex, vertex_id):
            lower_link.append(star_vertex)
    if len(upper_link) == 0:
        return MAXIMUM, None
    if len(lower_link) == 0:
        return MINIMUM, None
    upper_components = get_connected_components(mesh, upper_link)
    lower_components = get_connected_components(mesh, lower_link)
    if len(upper_components) == 2 and len(lower_components) == 2:
        return SADDLE, (upper_components, lower_components)
    if len(upper_components) == 3 and len(lower_components) == 3:
        return MONKEY_SADDLE, (upper_components, lower_components)
    return REGULAR, None

def random_sample_tri(v0, v1, v2):
    r1 = np.random.uniform(0, 1)
    r2 = np.random.uniform(0, 1)
    if (r1 + r2 > 1):
        r1 = 1 - r1
        r2 = 1 - r2
    p = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
    return p

def max_triangle_edge(v0, v1, v2):
    e = [v1 - v0, v2 - v0, v2 - v1]
    max_edge = -sys.float_info.max
    for i in range(3):
        max_edge = max(max_edge, np.linalg.norm(e[i]))
    return max_edge

def rotate2D(vector, degree):
    theta = (degree / 180.) * np.pi
    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta),  np.cos(theta)]])
    return rotate_matrix @ vector

# def sample_triangle()

