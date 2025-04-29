import itertools
from typing import List
import numpy as np
import queue
from pyvis.network import Network
import webbrowser


class Vertex:
    def __init__(self, id):
        self.id = id
        self.degree = 0
        self.in_degree = 0
        self.out_degree = 0
        self.in_edges = []
        self.out_edges = []
        self.pos_x = None
        self.pos_y = None


class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = np.zeros((n, n))  # adjacency matrix
        self.vertices: List[Vertex] = []
        for i in range(n):
            self.vertices.append(Vertex(i + 1))

    def add_edge(self, head, tail):
        u = self.vertices[head - 1]
        v = self.vertices[tail - 1]
        u.degree += 1
        u.out_degree += 1
        u.out_edges.append(v)
        v.degree += 1
        v.in_degree += 1
        v.in_edges.append(u)
        self.adj[head - 1, tail - 1] += 1

    def is_dynkin(self):
        cartan = np.zeros((self.n, self.n))
        for (i, j) in itertools.product(*[range(self.n)] * 2):
            if i == j:
                if self.adj[i, j] != 0:
                    return False
                cartan[i, j] = 2
                continue
            cartan[i, j] = -self.adj[i, j] - self.adj[j, i]

        for i in range(1, self.n + 1):
            if np.linalg.det(cartan[0:i, 0:i]) - 1e-6 <= 0:
                return False

        return True

    def find_coxeter(self):
        phi = np.identity(self.n)

        q = queue.Queue()
        for i in range(self.n):
            if self.vertices[i].in_degree == 0:
                q.put(self.vertices[i])

        while not q.empty():
            v = q.get()
            sv = np.identity(self.n)
            sv[v.id - 1, v.id - 1] = -1
            for u in v.in_edges + v.out_edges:
                sv[v.id - 1, u.id - 1] = 1
            phi = np.dot(sv, phi)
            for u in v.out_edges:
                u.in_degree -= 1
                if u.in_degree == 0:
                    q.put(u)

        for i in range(self.n):
            self.vertices[i].in_degree = len(self.vertices[i].in_edges)

        return phi

    def cal_pos_x(self):
        q = queue.Queue()
        q.put(self.vertices[0])
        self.vertices[0].pos_x = 0
        while not q.empty():
            v = q.get()
            for u in v.in_edges:
                if u.pos_x is not None:
                    continue
                u.pos_x = v.pos_x + 1
                q.put(u)

            for u in v.out_edges:
                if u.pos_x is not None:
                    continue
                u.pos_x = v.pos_x - 1
                q.put(u)

    def cal_pos_y(self):
        p = self.vertices[0]
        for i in range(self.n):
            if self.vertices[i].degree == 3:
                p = self.vertices[i]
                break

        p.pos_y = 0
        edges = p.in_edges + p.out_edges
        for u in edges:
            if u.degree == 1:
                u.pos_y = -1
                edges.remove(u)
                break

        down = edges[0]

        if len(edges) > 1:
            up = edges[1]
            up.pos_y = -2
            while up.degree > 1:
                for u in up.in_edges + up.out_edges:
                    if u.pos_y is None:
                        u.pos_y = up.pos_y - 1
                        up = u
                        break

        down.pos_y = 1
        while down.degree > 1:
            for u in down.in_edges + down.out_edges:
                if u.pos_y is None:
                    u.pos_y = down.pos_y + 1
                    down = u
                    break

    def cal_projective(self, i):
        alpha = np.zeros((self.n, 1))
        q = queue.Queue()
        q.put(self.vertices[i])
        while not q.empty():
            v = q.get()
            alpha[v.id - 1, 0] += 1
            for u in v.out_edges:
                q.put(u)
        return alpha


def is_nonnegetive(alpha, n):
    for i in range(n):
        if alpha[i, 0] + 1e-6 < 0:
            return False
    return True


def vec_to_str(vec, n):
    s = "["
    for i in range(n - 1):
        s += f"{int(vec[i, 0] + 1e-6)}, "
    s += f"{int(vec[n - 1, 0] + 1e-6)}]"
    return s


def main():
    # input
    n = int(input('Please input the number of vertices:\n'))
    print('Please input edges:')
    g = Graph(n)
    for i in range(n - 1):
        head, tail = list(map(int, input().split()))
        g.add_edge(head, tail)

    if not g.is_dynkin():
        print('This quiver is not Dynkin.')
        return

    coxeter = g.find_coxeter()

    g.cal_pos_x()
    g.cal_pos_y()
    min_x = 0
    min_y = 0
    for i in range(n):
        if g.vertices[i].pos_y < min_y:
            min_y = g.vertices[i].pos_y
        if g.vertices[i].pos_x < min_x:
            min_x = g.vertices[i].pos_x

    net = Network(height='600px', width='100%', directed=True)
    net.toggle_physics(False)
    state = [0 for _ in range(n)]
    last_state = [0 for _ in range(n)]
    vectors = [g.cal_projective(i) for i in range(n)]
    over = 0
    x = min_x
    while over < n:
        for i in range(n):
            if g.vertices[i].pos_x == x:
                state[i] = 1
                node_id = f'({i + 1},{x - min_x})'
                y = g.vertices[i].pos_y
                nvec = np.dot(coxeter, vectors[i])
                color = '#97c2fc'
                if last_state[i] == 0:
                    color = '#345485'
                    if not is_nonnegetive(nvec, n):
                        color = '#f2eedd'
                elif not is_nonnegetive(nvec, n):
                    color = '#b9d2d7'

                net.add_node(node_id,
                             label=node_id,
                             x=(x - min_x) * 100,
                             y=(y - min_y) * 100,
                             title=f'{vec_to_str(vectors[i], n)}',
                             color=color,
                             physics=False)

                for u in g.vertices[i].in_edges + g.vertices[i].out_edges:
                    if last_state[u.id - 1] == 1:
                        net.add_edge(f'({u.id},{x - min_x - 1})', node_id, color='#6b90cb')

                vectors[i] = nvec
                if is_nonnegetive(vectors[i], n):
                    g.vertices[i].pos_x += 2
                else:
                    over += 1

        last_state = state
        state = [last_state[i] for i in range(n)]
        x = x + 1

    net.write_html("ARQuiver.html")
    webbrowser.open("ARQuiver.html")


if __name__ == '__main__':
    main()
