from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.tri as mtri
from test_fun import *
from util import *
MIN_EDGE = 1e-5
MIN_GRAD = 0.01
h = [-5, 5]
v = [-5, 5]
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
points = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
mesh = Delaunay(points, incremental=True)
bad_criticals = set()
for i in range(len(mesh.points)):
    info = point_identification(mesh, i)
    if info[0] != REGULAR:
        grad_len = np.linalg.norm(grad_test_fun(mesh.points[i]))
        if grad_len > MIN_GRAD and ~on_boundary(mesh.points[i], h, v): # 不在边界上的bad criticals
            bad_criticals.add(i)

for iter in range(1000):
    samples = []
    for i in bad_criticals:
        neighbours = get_vertex_neighbours(i, mesh)
        for v in neighbours:
            length = np.linalg.norm(mesh.points[v] - mesh.points[i])
            if length > MIN_EDGE:
                sample = mesh.points[v] * 0.5 + mesh.points[i] * (1 - 0.5)
                samples.append(sample)
                
    if len(samples) == 0:
        print("Done")
        break
    
    samples = np.unique(samples, axis=0)
    start_id = len(mesh.points)
    mesh.add_points(samples)
    points_affected = set(get_vertex_neighours_from(start_id, mesh).tolist())
    for i in range(len(samples)):
        points_affected.add(start_id + i)
    for i in points_affected:
        info = point_identification(mesh, i)
        grad_len = np.linalg.norm(grad_test_fun(mesh.points[i]))
        if i in bad_criticals:
           if info[0] == REGULAR or grad_len <= MIN_GRAD:
               bad_criticals.remove(i)
        else:
            if info[0] != REGULAR and grad_len > MIN_GRAD:
                bad_criticals.add(i)
                
    print(len(bad_criticals))
    print(mesh.coplanar)
    # bads = mesh.points[list(bad_criticals)]
    # length = list(np.linalg.norm(i) for i in grad_test_fun(bad_criticals))
    # length.sort()
    # print(length[0])
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot()
    # ax1.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.simplices)
    # ax1.scatter(bads[:, 0], bads[:, 1], color="red")
    # plt.show()
    
    
    
    
    # samples = []
    # visited = np.zeros(len(mesh.simplices), dtype=np.bool_)
    # adj_tri = {}
    # for i in range(len(mesh.simplices)):
    #     tri = mesh.simplices[i]
    #     for j in range(3):
    #         if adj_tri.get(tri[j]) == None:
    #             adj_tri[tri[j]] = set()
    #         adj_tri[tri[j]].add(i)
    # for i in range(len(mesh.points)): 
    #     info = point_identification(mesh, i)
    #     if info[0] == REGULAR:
    #         min_dist = sys.float_info.max
    #         for j in critical_ids:
    #             dist = np.linalg.norm(mesh.points[i] - mesh.points[j])
    #             if dist < min_dist:
    #                 min_dist = dist
    #         # print(min_dist)
    #         if min_dist < 0.05 or min_dist > 0.3:
    #             # print("Close Regular")
    #             samples.extend(generate_sample(mesh, i, adj_tri, visited))
    #     else:
    #         grad_len = np.linalg.norm(grad_test_fun(mesh.points[i]))
    #         if grad_len > 0.1: # threshold
    #             # print("Bad Critical")
    #             samples.extend(generate_sample(mesh, i, adj_tri, visited))
    # mesh.add_points(samples)
    # print(len(samples))
    # plt.show()
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot()
    # ax1.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.simplices)
    # plt.show()





# for i in range(len(points)):
#     info = point_identification(mesh, i)
#     if info[0] != REGULAR:
#         grad_len = np.linalg.norm(grad_test_fun(points[i]))
#         if grad_len > threshold:
#             bad_critical_points.add(i)


# for iter in range(1000):
#     samples = []
#     for bad_critical_point in bad_critical_points:
#         neighbours = get_vertex_neighbours(bad_critical_point, mesh)
#         for v in neighbours:
#             length = np.linalg.norm(mesh.points[v] - mesh.points[bad_critical_point])
#             if length >= min_triangle_edge:
#                 t = 0.8
#                 sample = mesh.points[v] * t + mesh.points[bad_critical_point] * (1 - t)
#                 samples.append(sample)
              
#     if len(samples) == 0:
#         print("Done")
#         break
#     start_id = len(mesh.points)
#     mesh.add_points(samples)
#     points_affected = set(get_vertex_neighours_from(start_id, mesh).tolist())
#     for i in range(len(samples)):
#         points_affected.add(start_id + i)
#     for i in points_affected:
#         info = point_identification(mesh, i)
#         grad_len = np.linalg.norm(grad_test_fun(mesh.points[i]))
#         if i in bad_critical_points:
#            if info[0] == REGULAR or grad_len <= threshold:
#                bad_critical_points.remove(i)
#         else:
#             if info[0] != REGULAR and grad_len > threshold:
#                 bad_critical_points.add(i)
                
#     print(len(bad_critical_points))
#     bad_criticals = mesh.points[list(bad_critical_points)]
#     length = list(np.linalg.norm(i) for i in grad_test_fun(bad_criticals))
#     length.sort()
#     print(length[0])
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot()
    # ax1.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.simplices)
    # ax1.scatter(bad_criticals[:, 0], bad_criticals[:, 1], color="red")
    # plt.show()
    
    
# vertex_info = {} # vertex -> type (saddle, max, min, regular)
# vertex_info = {}
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(projection='3d')
# ax1.plot_trisurf(points[:, 0], points[:, 1], test_fun(points), cmap="viridis", alpha=0.5)
# for i in range(len(points)):
#     vertex_info[i] = point_identification(mesh, vertex_id=i)
#     color = None
#     label = None
#     if vertex_info[i][0] == MAXIMUM:
#         color = "red"
#     if vertex_info[i][0] == MINIMUM:
#         color = "blue"
#     if vertex_info[i][0] == SADDLE:
#         color = "green"
#     if vertex_info[i][0] == MONKEY_SADDLE:
#         color = "cyan"
#     if color != None:
#         ax1.scatter(points[i][0], points[i][1], test_fun(points[i]), color=color)
        
# plt.show()


# 样本的稠密程度
# 是否是critical point
# critical point可以弄一个比较尖的高斯 -> 范围小
# regular point可以弄一个比较平的高斯 -> 范围大
