from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from critical_detect import *
import sys

min_triangle_len = 1e-5
threshold = 0.01

def ackley_function(x, y, a=20, b=0.2, c=2 * np.pi):
    exp1 = np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    exp2 = np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return -a * exp1 - exp2 + a + np.exp(1)

def d_ackley_function(x, y, a=20, b=0.2, c=2 * np.pi):
    dfdx = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2))) * (-b * x / np.sqrt(2 * (x**2 + y**2))) \
        + np.exp(0.5 * (np.cos(c * x) + np.cos(c * y))) * 0.5 * c * np.sin(c * x)
    dfdy = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2))) * (-b * y / np.sqrt(2 * (x**2 + y**2))) \
        + np.exp(0.5 * (np.cos(c * x) + np.cos(c * y))) * 0.5 * c * np.sin(c * y)
    dfdx = dfdx.reshape(-1, 1)
    dfdy = dfdy.reshape(-1, 1)
    return np.concatenate((dfdx, dfdy), axis=1)

def ackey_new_dir(x, y, a=20, b=0.2, c=2 * np.pi):
    fxx = -((0.707107 * a * b *np.exp(-0.707107*b *np.sqrt(x**2 + y**2)) *x**2)/ np.power(x**2 + y**2, 1.5))
    -(0.5*a*b*b *np.exp(-0.707107*b*np.sqrt(x**2 + y**2)) *x**2) / (x**2 + y**2)
    +(0.707107*a *b *np.exp(-0.707107*b*np.sqrt(x**2 + y**2))) / np.sqrt(x**2 + y**2)
    + 0.5*c*c*np.exp(0.5*(np.cos(c*x) + np.cos(c*y)))*np.cos(c*x) 
    - 0.25* c*c *np.exp(0.5*(np.cos(c* x) + np.cos(c*y)))*np.power(np.sin(c*x), 2)
    
    fyy = -((0.707107 * a * b *np.exp(-0.707107*b *np.sqrt(x**2 + y**2)) *y**2)/ np.power(x**2 + y**2, 1.5))
    -((0.5*a*b**2 *np.exp(-0.707107*b*np.sqrt(x**2 + y**2)) *y**2) / (x**2 + y**2))
    +(0.707107*a *b *np.exp(-0.707107*b*np.sqrt(x**2 + y**2))) / np.sqrt(x**2 + y**2)
    + 0.5*c*c*np.exp(0.5*(np.cos(c*x) + np.cos(c*y)))*np.cos(c*y) 
    - 0.25* c*c *np.exp(0.5*(np.cos(c* x) + np.cos(c*y)))*np.power(np.sin(c*y), 2)
    
    fxy= -((0.707107*a *b *np.exp(-0.707107*b *np.sqrt(x**2 + y**2))*x *y)/np.power(x**2 + y**2, 1.5))
    -(0.5*a*b*b *np.exp(-0.707107*b *np.sqrt(x**2 + y**2))*x *y)/(x**2 + y**2) 
    -0.25*c*c*np.exp(0.5*(np.cos(c*x) + np.cos(c* y)))*np.sin(c*x)*np.sin(c *y)
    
    fx = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2))) * (-b * x / np.sqrt(2 * (x**2 + y**2))) \
        + np.exp(0.5 * (np.cos(c * x) + np.cos(c * y))) * 0.5 * c * np.sin(c * x)
    fy = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2))) * (-b * y / np.sqrt(2 * (x**2 + y**2))) \
        + np.exp(0.5 * (np.cos(c * x) + np.cos(c * y))) * 0.5 * c * np.sin(c * y)
    v1 = (-fx * fxx - fy * fxy).reshape(-1, 1)
    v2 = (-fx * fxy - fy * fyy).reshape(-1, 1)
    return np.concatenate((v1, v2), axis=1)


def fun1(x, y):
    return 1 / 4000 * (x**2 + y**2) - np.cos(x) * np.cos(y / np.sqrt(2)) + 1

def d_fun1(x, y):
    dfdx = x / 2000 + np.cos(y / np.sqrt(2)) * np.sin(x)
    dfdy = y / 2000 + (np.cos(x) * np.sin(y / np.sqrt(2))) / np.sqrt(2)
    dfdx = dfdx.reshape(-1, 1)
    dfdy = dfdy.reshape(-1, 1)
    return np.concatenate((dfdx, dfdy), axis=1)

def fun1_new_dir(x, y):
    fx = x / 2000 + np.cos(y / np.sqrt(2)) * np.sin(x)
    fy = y / 2000 + (np.cos(x) * np.sin(y / np.sqrt(2))) / np.sqrt(2)
    fxx = 1/2000 + np.cos(x) * np.cos(y/np.sqrt(2))
    fyy = 1/2000 + 1/2 * np.cos(x) *np.cos(y/np.sqrt(2))
    fxy = -((np.sin(x) * np.sin(y/np.sqrt(2)))/np.sqrt(2))
    v1 = (-fx * fxx - fy * fxy).reshape(-1, 1)
    v2 = (-fx * fxy - fy * fyy).reshape(-1, 1)
    return np.concatenate((v1, v2), axis=1)

def initial_function(x, y):
    # return np.sin(x) + np.cos(y)
    # return ackley_function(x, y)
    return fun1(x, y)

def initial_function_grad(x, y):
    # dfdx = np.cos(x).reshape(-1, 1)
    # dfdy = -np.sin(y).reshape(-1, 1)
    # return np.concatenate((dfdx, dfdy), axis=1)
    # return d_ackley_function(x, y)
    return d_fun1(x, y)

def new_direction(x, y):
    v1 = np.sin(x) * np.cos(x).reshape(-1, 1)
    v2 = -np.sin(y) * np.cos(y).reshape(-1, 1)
    return np.concatenate((v1, v2), axis=1)
    # return ackey_new_dir(x, y)
    
###########################################################
h = [-5, 5]
v = [-5, 5]
rng = np.random.default_rng()
engine = qmc.PoissonDisk(d=2, radius=0.01, seed=rng)
samples = engine.fill_space()
boundary = []
for i in np.linspace(0, 1, num=50, endpoint=True):
    boundary.extend([[0, i], [1, i], [i, 0], [i, 1]])
samples = np.concatenate((samples, boundary))
samples = np.unique(samples, axis=0, return_index=False, return_counts=False)
samples[:, 0] = samples[:, 0] * (h[1] - h[0]) + h[0]
samples[:, 1] = samples[:, 1] * (v[1] - v[0]) + v[0]

########################################################### Refine Criticals
def check_boundary(p):
    return abs(p[0]-h[0])<=1e-5 or abs(p[0] -h[1]) <=1e-5 or abs(p[1]- v[0])<=1e-5 or abs(p[1] - v[1])<=1e-5

def find_max_edge_len(triangle):
    length = []
    for i in range(3):
        p1 = triangle[i]
        p2 = triangle[(i + 1) % 3]
        length.append(np.linalg.norm(p1 - p2))
    return max(length)

def local_sampler(new_samples, triangle, depth=1):
    if find_max_edge_len(triangle) <= min_triangle_len:
        print("Triangle too small, draw 0 samples")
        return
    if depth == 0:
        return
    bary_center = np.array([0, 0])
    for i in triangle:
        bary_center = bary_center + i
    bary_center = bary_center / 3
    new_samples.append(bary_center)
    for i in range(3):
        v0 = np.array(triangle[i])
        v1 = np.array(triangle[(i + 1) % 3])
        mid = (v0 + v1) / 2
        local_sampler(new_samples, [bary_center, v0, mid], depth - 1)
        local_sampler(new_samples, [bary_center, v1, mid], depth - 1)

def ray_line_intersection(p, r, q, s):
    t_up = np.cross(q - p, s)
    down = np.cross(r, s)
    u_up = np.cross(q - p, r)
    if down == 0:
        return False, -1
    u = u_up / down
    t = t_up / down
    if u >= 0 and u <= 1 and t >= 0:
        return True, t
    return False, -1


def trace_min(u, path, vis=None, junctions=None, is_critical=None, refine=True):
    while True:
        if refine:
            if is_critical[u] == 0 and vis[u] == 1:
                junctions.append(u)
            vis[u] = 1
        path.append(u)
        minv = sys.float_info.max
        next_node = -1
        nu = list(graph[u].keys())
        for v in nu:
            if values[v] < minv or (values[v] == minv and v < u):
                minv = values[v]
                next_node = v
        if minv < values[u] or (minv == values[u] and next_node < u):
            u = next_node
        else:
            break

def trace_max(u, path, vis=None, junctions=None, is_critical=None, refine=True):
    while True:
        if refine:
            if is_critical[u] == 0 and vis[u] == 1:
                junctions.append(u)
            vis[u] = 1
        path.append(u)
        maxv = -sys.float_info.max
        next_node = -1
        nu = list(graph[u].keys())
        for v in nu:
            if values[v] > maxv or (values[v] == maxv and v > u):
                maxv = values[v]
                next_node = v
        if maxv > values[u] or (maxv == values[u] and next_node > u):
            u = next_node
        else:
            break

iteration = 0
while iteration < 100:
    mesh = mtri.Triangulation(samples[:, 0], samples[:, 1])
    values = initial_function(samples[:, 0], samples[:, 1])
    graph, adj_triangles = graph_constructor(mesh)
    min_points, max_points, saddle_points, links = critical_detector(graph, values)
    
    sampled = np.zeros(len(mesh.triangles))
    is_critical = np.zeros(len(samples))
    new_samples = []
    for i in min_points:
        p = samples[i]
        grad_len = np.linalg.norm(initial_function_grad(p[0], p[1]))
        is_critical[i] = 1
        if ~check_boundary(p) and grad_len > threshold:
            for t_id, v_id in adj_triangles[i]:
                if sampled[t_id] == 0:
                    sampled[t_id] = 1
                    tri = np.array(list(map(lambda x: samples[x], mesh.triangles[t_id])))
                    local_sampler(new_samples, tri)
    for i in max_points:
        p = samples[i]
        grad_len = np.linalg.norm(initial_function_grad(p[0], p[1]))
        is_critical[i] = 1
        if ~check_boundary(p) and grad_len > threshold:
            for t_id, v_id in adj_triangles[i]:
                if sampled[t_id] == 0:
                    sampled[t_id] = 1
                    tri = np.array(list(map(lambda x: samples[x], mesh.triangles[t_id])))
                    local_sampler(new_samples, tri)
    for i in saddle_points:
        p = samples[i]
        grad_len = np.linalg.norm(initial_function_grad(p[0], p[1]))
        is_critical[i] = 1
        if ~check_boundary(p) and grad_len > threshold:
            for t_id, v_id in adj_triangles[i]:
                if sampled[t_id] == 0:
                    sampled[t_id] = 1
                    tri = np.array(list(map(lambda x: samples[x], mesh.triangles[t_id])))
                    local_sampler(new_samples, tri)
    
    cp_samples = len(new_samples)
    visited = np.zeros(len(samples))
    junctions = []
    print(len(junctions))
    for u in saddle_points:
        visited[u] = 1
        LK, UK = links[u]
        for _, cp in LK.items():
            idx = min(cp, key=lambda x: values[x])
            path = []
            trace_min(idx, path, visited, junctions, is_critical)

        # enumerate upper links connected components/wedges
        for _, cp in UK.items():
            idx = max(cp, key=lambda x: values[x])
            path = []
            trace_max(idx, path, visited, junctions, is_critical)
            
    for i in junctions:
        for t_id, v_id in adj_triangles[i]:
            if sampled[t_id] == 0:
                sampled[t_id] = 1
                tri = np.array(list(map(lambda x: samples[x], mesh.triangles[t_id])))
                local_sampler(new_samples, tri)
    
    print(f"Samples drawn for critical points = {cp_samples}")
    print(f"Samples drawn for junction points = {len(new_samples) - cp_samples}")
    
    if len(new_samples) == 0:
        print("Done.")
        break
    samples = np.append(samples, new_samples).reshape(-1, 2)
    iteration += 1
    print(f"Iteration: {iteration}, Samples: {len(new_samples)}")

mesh = mtri.Triangulation(samples[:, 0], samples[:, 1])
values = initial_function(samples[:, 0], samples[:, 1])
graph, adj_triangles = graph_constructor(mesh)
min_points, max_points, saddle_points, links = critical_detector(graph, values)

min_values = np.array(list(map(lambda x: values[x], min_points)))
max_values = np.array(list(map(lambda x: values[x], max_points)))
saddle_values = np.array(list(map(lambda x: values[x], saddle_points)))
min_vertices = np.array(list(map(lambda x: samples[x], min_points)))
max_vertices = np.array(list(map(lambda x: samples[x], max_points)))
saddle_vertices = np.array(list(map(lambda x: samples[x], saddle_points)))

grad_min = initial_function_grad(min_vertices[:, 0], min_vertices[:, 1])
grad_max = initial_function_grad(max_vertices[:, 0], max_vertices[:, 1])
grad_saddle = initial_function_grad(saddle_vertices[:, 0], saddle_vertices[:, 1])

grad_min_len = np.array(list(map(lambda x: np.linalg.norm(x), grad_min)))
grad_max_len = np.array(list(map(lambda x: np.linalg.norm(x), grad_max)))
grad_saddle_len = np.array(list(map(lambda x: np.linalg.norm(x), grad_saddle)))
print("total sample budget = ", len(samples))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title("After Refinement")
ax1.triplot(mesh, color="black", alpha=0.2)
ax1.scatter(min_vertices[:, 0], min_vertices[:, 1], color="blue", label="min")
ax1.scatter(max_vertices[:, 0], max_vertices[:, 1], color="red", label="max")
ax1.scatter(saddle_vertices[:, 0], saddle_vertices[:, 1], color="green", label="saddle")
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.set_title(f"Residue")
# ax = plt.axes(projection='3d')
ax2.bar3d(
    max_vertices[:, 0],
    max_vertices[:, 1],
    0,
    0.05,
    0.05,
    grad_max_len,
    color="red",
    label="max",
)
ax2.bar3d(
    min_vertices[:, 0],
    min_vertices[:, 1],
    0,
    0.05,
    0.05,
    grad_min_len,
    color="blue",
    label="min",
)
ax2.bar3d(
    saddle_vertices[:, 0],
    saddle_vertices[:, 1],
    0,
    0.05,
    0.05,
    grad_saddle_len,
    color="green",
    label="saddle",
)
plt.legend()
plt.show()


fig3 = plt.figure()
ax3 = fig3.add_subplot(projection='3d')
ax3.set_title(f"Function Plot w/ integral lines")
ax3.plot_trisurf(samples[:, 0], samples[:, 1], values, cmap="viridis", alpha=0.5)
ax3.scatter(min_vertices[:, 0], min_vertices[:, 1], min_values, color="blue", label="min")
ax3.scatter(max_vertices[:, 0], max_vertices[:, 1], max_values, color="red", label="max")
ax3.scatter(saddle_vertices[:, 0], saddle_vertices[:, 1], saddle_values, color="green", label="saddle")


fig4 = plt.figure()
ax4 = fig4.add_subplot()
ax4.set_title(f"2D Mesh w/ integral lines")
ax4.triplot(mesh, color="black", alpha=0.2)
ax4.scatter(min_vertices[:, 0], min_vertices[:, 1], color="blue", label="min")
ax4.scatter(max_vertices[:, 0], max_vertices[:, 1], color="red", label="max")
ax4.scatter(saddle_vertices[:, 0], saddle_vertices[:, 1], color="green", label="saddle")

visited = np.zeros(len(samples))
intline_D = {}
for u in saddle_points:
    visited[u] = 1
    LK, UK = links[u]
    intline_D[u] = [[], []]  # min path x 2, max path x 2
    for _, cp in LK.items():
        idx = min(cp, key=lambda x: values[x])
        path = []
        trace_min(idx, path, refine=False)
        intline_D[u][0].append(path)

    # enumerate upper links connected components/wedges
    for _, cp in UK.items():
        idx = max(cp, key=lambda x: values[x])
        path = []
        trace_max(idx, path, refine=False)
        intline_D[u][1].append(path)
        

for saddle, paths in intline_D.items():
    min_path = paths[0]  # min path
    max_path = paths[1]  # max path
    for path in min_path:
        new_path = [saddle, *path]
        vs = np.array(list(map(lambda x: samples[x], new_path)))
        vals = np.array(list(map(lambda x: values[x], new_path)))
        ax3.plot(vs[:, 0], vs[:, 1], vals, color="blue", linestyle="--")
        ax4.plot(vs[:, 0], vs[:, 1], color="blue", linestyle="--")
    for path in max_path:
        new_path = [saddle, *path]
        vs = np.array(list(map(lambda x: samples[x], new_path)))
        vals = np.array(list(map(lambda x: values[x], new_path)))
        ax3.plot(vs[:, 0], vs[:, 1], vals, color="red")
        ax4.plot(vs[:, 0], vs[:, 1], color="red")

plt.show()

# if ~check_boundary(p) and hit_record[0] == False:
#     print("There Must Be One Line Being Hit For Inner Points.")
#     plt.figure()
#     plt.title("Local Sample")
#     plt.triplot(mesh, color="black")
#     plt.scatter(p[0], p[1], color="red")
#     plt.show()