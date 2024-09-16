from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.tri as mtri
from sklearn.preprocessing import normalize

def critical_detector(graph, values):
    def find(x, P):
        if P[x] == x:
            return x
        else:
            P[x] = find(P[x], P)
            return P[x]

    def merge(x, y, P):
        a = find(x, P)
        b = find(y, P)
        if not a == b:
            P[b] = a

    def get_lower_link(u):  # return the lower link for this vertex: [[], [], ...]
        nu = list(graph[u].keys())
        val_u = values[u]
        lower_link = [] 
        for i in range(len(nu)):
            v = nu[i]
            val_v = values[v]
            if val_v < val_u:
                lower_link.append(i)
            elif val_v == val_u:
                if v < u:
                    lower_link.append(i)

        count = len(lower_link)

        P = []
        for i in range(count):
            P.append(i)

        for i in range(count):
            for j in range(i + 1, count):
                va, vb = nu[lower_link[i]], nu[lower_link[j]]
                if graph[va].get(vb) != None:
                    merge(i, j, P)

        LK = {}

        for i in range(count):
            set_id = find(i, P)
            if LK.get(set_id) == None:
                LK[set_id] = []
            LK[set_id].append(nu[lower_link[i]])

        return LK


    def get_upper_link(u):
        nu = list(graph[u].keys())
        val_u = values[u]
        upper_link = []
        for i in range(len(nu)):
            v = nu[i]
            val_v = values[v]
            if val_v > val_u:
                upper_link.append(i)
            elif val_v == val_u:
                if v > u:
                    upper_link.append(i)

        count = len(upper_link)
        P = []
        for i in range(count):
            P.append(i)

        for i in range(count):
            for j in range(i + 1, count):
                va, vb = nu[upper_link[i]], nu[upper_link[j]]
                if graph[va].get(vb) != None:
                    merge(i, j, P)

        UK = {}

        for i in range(count):
            set_id = find(i, P)
            if UK.get(set_id) == None:
                UK[set_id] = []
            UK[set_id].append(nu[upper_link[i]])

        return UK

    def fetch_criticals(links):
        min_points = []
        max_points = []
        saddle_points = []

        for i in graph.keys():
            llk_ccpn = len(links[i][0].keys())
            ulk_ccpn = len(links[i][1].keys())
            if llk_ccpn == 0 and ulk_ccpn == 1:
                # min
                min_points.append(i)
            elif ulk_ccpn == 0 and llk_ccpn == 1:
                # max
                max_points.append(i)
            elif ulk_ccpn == 2 and llk_ccpn == 2:
                # saddle
                saddle_points.append(i)
            elif ulk_ccpn == 3 and llk_ccpn == 3:
                saddle_points.append(i)
                print("find a monkey saddle (add to saddle_points) !")
        return min_points, max_points, saddle_points
                
    links = {}
    for u in graph.keys():
        llk, ulk = get_lower_link(u), get_upper_link(u)
        links[u] = (llk, ulk)
     
    criticals = fetch_criticals(links)
    return *criticals, links
        

def graph_constructor(mesh):
    graph = {}
    adj_triangles = {}
    def connect(va, vb):
        if graph.get(va) == None:
            graph[va] = {}
        graph[va][vb] = True
    for idx, t in enumerate(mesh.triangles):
        connect(t[0], t[1])
        connect(t[1], t[0])
        connect(t[0], t[2])
        connect(t[2], t[0])
        connect(t[1], t[2])
        connect(t[2], t[1])
        for i in range(3):
            if adj_triangles.get(t[i]) == None:
                adj_triangles[t[i]] = []
            adj_triangles[t[i]].append((idx, i))
        
    return graph, adj_triangles