"""
agents.py
---------
Três agentes que competem no mesmo labirinto, cada um com uma estratégia
diferente para chegar ao objetivo. Todos compartilham a mesma interface
(BaseAgent), então o jogo pode tratá-los de forma intercambiável.

1. AStarAgent      -> busca heurística A* (estado, objetivo, heurística)
2. QLearningAgent  -> aprendizado por reforço (tabela Q treinada antes do jogo)
3. GeneticAgent    -> algoritmo genético (evolui uma sequência de ações)
"""

import heapq
import random


MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # cima, baixo, esquerda, direita
N_ACTIONS = len(MOVES)


class BaseAgent:
    def __init__(self, name, color, short_label=None):
        self.name = name
        self.short_label = short_label or name
        self.color = color
        self.pos = None
        self.finished = False
        self.steps_taken = 0

    def prepare_round(self, maze, spawn):
        """Chamado no início de cada rodada, com um novo ponto de spawn."""
        self.pos = spawn
        self.spawn = spawn
        self.finished = False
        self.steps_taken = 0

    def step(self, maze):
        """Executa um passo no labirinto. Cada subclasse define sua estratégia."""
        raise NotImplementedError


# ======================================================================
# 1) AGENTE HEURÍSTICO -- Busca A*
# ======================================================================
class AStarAgent(BaseAgent):
    """
    Estado: posição (linha, coluna). Objetivo: chegar em maze.goal.
    Heurística: distância de Manhattan (admissível, pois cada passo custa 1
    e nunca é possível "cortar caminho" além do que ela prevê).
    O caminho inteiro é calculado uma vez em prepare_round(); step() apenas
    o percorre.
    """

    def __init__(self, color):
        super().__init__("Busca Heurística (A*)", color, short_label="Heurística")
        self.path = []
        self.path_index = 0

    def prepare_round(self, maze, spawn):
        super().prepare_round(maze, spawn)
        self.path = self._a_star(maze, spawn, maze.goal)
        self.path_index = 0

    @staticmethod
    def _heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _a_star(self, maze, start, goal):
        counter = 0  # desempate estável no heap (evita comparar tuplas com posições)
        open_heap = [(self._heuristic(start, goal), counter, start)]
        came_from = {}
        g_score = {start: 0}
        visited = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return self._reconstruct(came_from, current)

            for n in maze.neighbors(current):
                if not maze.passable(current, n) or n in visited:
                    continue
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(n, float("inf")):
                    g_score[n] = tentative_g
                    came_from[n] = current
                    counter += 1
                    heapq.heappush(open_heap, (tentative_g + self._heuristic(n, goal), counter, n))

        return [start]  # objetivo inalcançável (não deve ocorrer em labirinto conectado)

    @staticmethod
    def _reconstruct(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def step(self, maze):
        if self.finished:
            return
        if self.path_index < len(self.path) - 1:
            self.path_index += 1
            self.pos = self.path[self.path_index]
            self.steps_taken += 1
        if self.pos == maze.goal:
            self.finished = True


# ======================================================================
# 2) AGENTE DE APRENDIZADO POR REFORÇO -- Q-Learning
# ======================================================================
class QLearningAgent(BaseAgent):
    """
    Estado: posição. Ação: {cima, baixo, esquerda, direita}.
    Recompensa: +100 no objetivo, -1 por passo, -5 ao esbarrar em parede.
    Atualização (Bellman): Q(s,a) += alpha * (r + gamma * max Q(s') - Q(s,a))

    O agente é TREINADO offline (train()) antes da 1ª rodada, a partir de
    todos os pontos de spawn possíveis. Durante o jogo ele age de forma
    quase-gulosa (pequena chance de exploração), mostrando o que aprendeu.
    """

    def __init__(self, color, alpha=0.2, gamma=0.9, epsilon_train=0.25,
                 epsilon_play=0.05, train_episodes=6000):
        super().__init__("Aprendizado por Reforço (Q-Learning)", color, short_label="Reforço")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_train = epsilon_train
        self.epsilon_play = epsilon_play
        self.train_episodes = train_episodes
        self.q_table = {}
        self.rng = random.Random()
        self.trained = False

    def _get_q(self, state):
        return self.q_table.setdefault(state, [0.0] * N_ACTIONS)

    def _apply_move(self, maze, pos, action):
        """Aplica uma ação e devolve (nova_posição, recompensa)."""
        dr, dc = MOVES[action]
        candidate = (pos[0] + dr, pos[1] + dc)
        if maze.in_bounds(candidate) and maze.passable(pos, candidate):
            reward = 100.0 if candidate == maze.goal else -1.0
            return candidate, reward
        return pos, -5.0  # bateu na parede: fica parado, penalidade

    def train(self, maze, spawn_pool):
        """Fase de treinamento offline (roda uma vez antes da 1ª rodada)."""
        max_steps = maze.size * 8
        for _ in range(self.train_episodes):
            pos = self.rng.choice(spawn_pool)
            steps = 0
            while pos != maze.goal and steps < max_steps:
                q = self._get_q(pos)
                if self.rng.random() < self.epsilon_train:
                    action = self.rng.randrange(N_ACTIONS)
                else:
                    action = max(range(N_ACTIONS), key=lambda a: q[a])

                new_pos, reward = self._apply_move(maze, pos, action)
                next_q = self._get_q(new_pos)
                q[action] += self.alpha * (reward + self.gamma * max(next_q) - q[action])
                pos = new_pos
                steps += 1
        self.trained = True

    def prepare_round(self, maze, spawn):
        super().prepare_round(maze, spawn)
        self.best_distance_seen = maze.distances.get(spawn, 10 ** 6)
        self.stall_counter = 0

    def step(self, maze):
        if self.finished:
            return

        # Como a rodada só termina quando TODOS os agentes chegam, o agente
        # não pode ficar preso oscilando entre duas células por causa de
        # valores de Q empatados. Se ficar muito tempo sem se aproximar do
        # objetivo, aumentamos temporariamente a exploração.
        effective_epsilon = self.epsilon_play if self.stall_counter <= 60 else 0.6

        q = self._get_q(self.pos)
        if self.rng.random() < effective_epsilon:
            action = self.rng.randrange(N_ACTIONS)
        else:
            best_value = max(q)
            best_actions = [a for a in range(N_ACTIONS) if q[a] == best_value]
            action = self.rng.choice(best_actions)  # desempate aleatório evita ciclos

        self.pos, _ = self._apply_move(maze, self.pos, action)
        self.steps_taken += 1

        current_distance = maze.distances.get(self.pos, self.best_distance_seen)
        if current_distance < self.best_distance_seen:
            self.best_distance_seen = current_distance
            self.stall_counter = 0
        else:
            self.stall_counter += 1

        if self.pos == maze.goal:
            self.finished = True


# ======================================================================
# 3) AGENTE DE ALGORITMO GENÉTICO
# ======================================================================
class GeneticAgent(BaseAgent):
    """
    Indivíduo: sequência fixa de ações (genes em 0..3).
    Aptidão: quanto mais perto do objetivo o indivíduo termina (distância
    BFS real) e quanto menos passos usar, maior a aptidão -- com um grande
    bônus por realmente alcançar o objetivo.

    Como o spawn muda a cada rodada, a população EVOLUI DE NOVO no início
    de cada rodada (prepare_round), mostrando seleção, cruzamento e
    mutação "ao vivo". Se não convergir dentro do orçamento normal de
    gerações, tentamos de novo com cromossomos maiores; como última rede
    de segurança, completamos o trajeto final com o caminho mais curto
    (BFS) -- o que fica registrado em used_rescue_fallback.
    """

    def __init__(self, color, population_size=100, generations=150,
                 mutation_rate=0.06, elite_fraction=0.1):
        super().__init__("Algoritmo Genético", color, short_label="Genético")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.rng = random.Random()
        self.best_path_cells = []
        self.path_index = 0
        self.last_fitness_history = []  # para depuração/relatório
        self.used_rescue_fallback = False  # True se precisou da rede de segurança

    def prepare_round(self, maze, spawn):
        super().prepare_round(maze, spawn)
        self.used_rescue_fallback = False

        chromo_len = maze.size * 3
        generations = self.generations
        cells, history = None, []

        for _attempt in range(5):
            best_chromo, history = self._evolve(maze, spawn, chromo_len, generations)
            cells = self._simulate(maze, spawn, best_chromo)
            if cells[-1] == maze.goal:
                break
            # Não convergiu: dá mais "material genético" e mais tempo.
            chromo_len = int(chromo_len * 1.6)
            generations = int(generations * 1.4)

        if cells[-1] != maze.goal:
            rescue = maze.shortest_path(cells[-1], maze.goal)
            if rescue:
                cells = cells + rescue[1:]
                self.used_rescue_fallback = True

        self.last_fitness_history = history
        self.best_path_cells = cells
        self.path_index = 0

    def _random_chromosome(self, length):
        return [self.rng.randrange(N_ACTIONS) for _ in range(length)]

    def _simulate(self, maze, spawn, chromosome):
        """Executa um cromossomo passo a passo e devolve a lista de células visitadas."""
        pos = spawn
        cells = [pos]
        for gene in chromosome:
            dr, dc = MOVES[gene]
            candidate = (pos[0] + dr, pos[1] + dc)
            if maze.in_bounds(candidate) and maze.passable(pos, candidate):
                pos = candidate
            cells.append(pos)
            if pos == maze.goal:
                break
        return cells

    def _fitness(self, maze, spawn, chromosome):
        cells = self._simulate(maze, spawn, chromosome)
        final = cells[-1]
        steps_used = len(cells) - 1
        dist_to_goal = maze.distances.get(final, maze.size * 2)
        score = -dist_to_goal * 10 - steps_used * 0.05
        if final == maze.goal:
            score += 2000 - steps_used
        return score

    def _tournament(self, ranked, k=5):
        """Seleção por torneio: sorteia k indivíduos e devolve o melhor deles.
        `ranked` já está ordenado do melhor para o pior fitness."""
        idx = sorted(self.rng.sample(range(len(ranked)), min(k, len(ranked))))
        return ranked[idx[0]][0]

    def _crossover(self, parent1, parent2):
        if len(parent1) < 2:
            return parent1[:]
        point = self.rng.randrange(1, len(parent1))
        return parent1[:point] + parent2[point:]

    def _mutate(self, chromosome):
        return [self.rng.randrange(N_ACTIONS) if self.rng.random() < self.mutation_rate else gene
                for gene in chromosome]

    def _evolve(self, maze, spawn, chromo_len, generations):
        population = [self._random_chromosome(chromo_len) for _ in range(self.population_size)]
        best_overall, best_fitness_overall = None, float("-inf")
        history = []

        for _gen in range(generations):
            ranked = sorted(
                ((c, self._fitness(maze, spawn, c)) for c in population),
                key=lambda pair: pair[1],
                reverse=True,
            )
            history.append(ranked[0][1])
            if ranked[0][1] > best_fitness_overall:
                best_fitness_overall = ranked[0][1]
                best_overall = ranked[0][0]

            n_elite = max(2, int(self.population_size * self.elite_fraction))
            next_gen = [c for c, _ in ranked[:n_elite]]  # elitismo: os melhores passam direto

            while len(next_gen) < self.population_size:
                parent1 = self._tournament(ranked)
                parent2 = self._tournament(ranked)
                child = self._mutate(self._crossover(parent1, parent2))
                next_gen.append(child)

            population = next_gen

        return best_overall, history

    def step(self, maze):
        if self.finished:
            return
        if self.path_index < len(self.best_path_cells) - 1:
            self.path_index += 1
            self.pos = self.best_path_cells[self.path_index]
            self.steps_taken += 1
        if self.pos == maze.goal:
            self.finished = True