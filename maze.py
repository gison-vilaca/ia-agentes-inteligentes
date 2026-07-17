"""
maze.py
-------
Geração do labirinto e escolha dos pontos de spawn "justos".

Ideia central: o labirinto é um grafo de células, onde cada aresta aberta
é um movimento válido. A distância entre duas células é sempre calculada
por BFS (busca em largura), que garante o menor número de passos.

Justiça dos spawns: olhamos todas as células da borda, calculamos a
distância de cada uma até o objetivo, e escolhemos um grupo de células
que tenham TODAS a mesma distância. Assim, não importa em qual desses
pontos um agente nasça, o caminho mais curto até o centro tem o mesmo
tamanho — a aleatoriedade fica só em "quem nasce onde", não em "quem
começa mais perto".
"""

import random
from collections import deque, defaultdict


MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # cima, baixo, esquerda, direita


class Maze:
    def __init__(self, size=17, extra_connections=0.15, seed=None, n_spawns=4):
        if size % 2 == 0:
            size += 1  # tamanho ímpar garante um centro exato
        self.size = size
        self.rng = random.Random(seed)

        # Arestas abertas (passáveis): cada uma é um frozenset({a, b})
        # entre duas células vizinhas na grade.
        self.open_edges = set()
        self._generate_perfect_maze()
        self._add_loops(extra_connections)

        self.goal = (size // 2, size // 2)
        self.distances = self._bfs_distances(self.goal)
        self.spawn_points, self.spawn_distance = self._select_fair_spawns(n_spawns)

    # ------------------------------------------------------------------
    # Geometria básica do grid
    # ------------------------------------------------------------------
    def in_bounds(self, cell):
        r, c = cell
        return 0 <= r < self.size and 0 <= c < self.size

    def neighbors(self, cell):
        r, c = cell
        for dr, dc in MOVES:
            n = (r + dr, c + dc)
            if self.in_bounds(n):
                yield n

    def passable(self, a, b):
        return frozenset((a, b)) in self.open_edges

    # ------------------------------------------------------------------
    # Geração do labirinto
    # ------------------------------------------------------------------
    def _generate_perfect_maze(self):
        """Recursive backtracker: gera um labirinto "perfeito" (existe
        exatamente um caminho entre quaisquer duas células)."""
        start = (0, 0)
        visited = {start}
        stack = [start]
        while stack:
            current = stack[-1]
            unvisited = [n for n in self.neighbors(current) if n not in visited]
            if unvisited:
                nxt = self.rng.choice(unvisited)
                self.open_edges.add(frozenset((current, nxt)))
                visited.add(nxt)
                stack.append(nxt)
            else:
                stack.pop()

    def _all_edges(self):
        edges = set()
        for r in range(self.size):
            for c in range(self.size):
                cell = (r, c)
                for n in self.neighbors(cell):
                    edges.add(frozenset((cell, n)))
        return edges

    def _add_loops(self, fraction):
        """Abre uma fração extra de paredes fechadas, criando ciclos e
        caminhos alternativos (um labirinto perfeito só tem um caminho
        entre dois pontos; aqui adicionamos opções extras)."""
        if fraction <= 0:
            return
        closed = list(self._all_edges() - self.open_edges)
        self.rng.shuffle(closed)
        n_open = int(len(closed) * fraction)
        self.open_edges.update(closed[:n_open])

    # ------------------------------------------------------------------
    # Busca em largura (BFS)
    # ------------------------------------------------------------------
    def _bfs_distances(self, start):
        """Distância (em passos) do ponto `start` até cada célula alcançável."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for n in self.neighbors(cur):
                if n not in dist and self.passable(cur, n):
                    dist[n] = dist[cur] + 1
                    queue.append(n)
        return dist

    def shortest_path(self, start, goal):
        """Caminho mais curto (BFS) entre duas células, ou None se não existir."""
        if start == goal:
            return [start]
        prev = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == goal:
                break
            for n in self.neighbors(cur):
                if n not in prev and self.passable(cur, n):
                    prev[n] = cur
                    queue.append(n)
        if goal not in prev:
            return None

        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Seleção justa dos pontos de spawn
    # ------------------------------------------------------------------
    def _border_cells(self):
        cells = set()
        for i in range(self.size):
            cells.add((0, i))
            cells.add((self.size - 1, i))
            cells.add((i, 0))
            cells.add((i, self.size - 1))
        cells.discard(self.goal)
        return list(cells)

    def _select_fair_spawns(self, k):
        """Agrupa as células da borda por distância até o objetivo e
        escolhe `k` delas dentro do grupo mais distante que tenha opções
        suficientes (spawns difíceis, mas sempre igualmente distantes)."""
        groups = defaultdict(list)
        for cell in self._border_cells():
            d = self.distances.get(cell)
            if d is not None:
                groups[d].append(cell)

        candidate_distances = sorted(
            (d for d, cells in groups.items() if len(cells) >= k), reverse=True
        )
        chosen_d = candidate_distances[0] if candidate_distances else max(groups, key=lambda d: len(groups[d]))

        pool = groups[chosen_d]
        chosen = self._spread_selection(pool, min(k, len(pool)))
        return chosen, chosen_d

    def _spread_selection(self, pool, k):
        """Escolhe k células do pool tentando maximizar a distância entre
        elas, para que os spawns fiquem espalhados pelo labirinto."""
        if len(pool) <= k:
            return list(pool)
        chosen = [self.rng.choice(pool)]
        remaining = [p for p in pool if p != chosen[0]]
        while len(chosen) < k and remaining:
            best = max(remaining, key=lambda p: min(self._manhattan(p, c) for c in chosen))
            chosen.append(best)
            remaining.remove(best)
        return chosen

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])