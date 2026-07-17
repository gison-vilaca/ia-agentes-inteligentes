"""
maze.py
-------
Geração do labirinto e escolha dos pontos de partida "justos".

Ideia central: o labirinto é um grafo de células, onde cada aresta aberta
é um movimento válido. A distância entre duas células é sempre calculada
por BFS (busca em largura), que garante o menor número de passos.

Justiça dos pontos de partida: olhamos as células da borda, calculamos a
distância de cada uma até o objetivo, e escolhemos um grupo com a MESMA
distância. Assim, não importa em qual desses pontos um agente nasça, o
caminho mais curto até o centro tem o mesmo tamanho.
"""

import random
from collections import deque, defaultdict


MOVIMENTOS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # cima, baixo, esquerda, direita


class Labirinto:
    def __init__(self, tamanho=17, conexoes_extras=0.15, semente=None, n_pontos_partida=4):
        if tamanho % 2 == 0:
            tamanho += 1  # tamanho ímpar garante um centro exato
        self.tamanho = tamanho
        self.aleatorio = random.Random(semente)

        # Arestas abertas (passáveis): cada uma é um frozenset({a, b})
        # entre duas células vizinhas na grade.
        self.arestas_abertas = set()
        self._gerar_labirinto_perfeito()
        self._adicionar_atalhos(conexoes_extras)

        self.objetivo = (tamanho // 2, tamanho // 2)
        self.distancias = self._distancias_por_bfs(self.objetivo)
        self.pontos_partida, self.distancia_partida = self._selecionar_pontos_justos(n_pontos_partida)

    # ------------------------------------------------------------------
    # Geometria básica do grid
    # ------------------------------------------------------------------
    def dentro_dos_limites(self, celula):
        linha, coluna = celula
        return 0 <= linha < self.tamanho and 0 <= coluna < self.tamanho

    def vizinhos(self, celula):
        linha, coluna = celula
        for dl, dc in MOVIMENTOS:
            vizinho = (linha + dl, coluna + dc)
            if self.dentro_dos_limites(vizinho):
                yield vizinho

    def passavel(self, a, b):
        return frozenset((a, b)) in self.arestas_abertas

    # ------------------------------------------------------------------
    # Geração do labirinto
    # ------------------------------------------------------------------
    def _gerar_labirinto_perfeito(self):
        """Recursive backtracker: gera um labirinto "perfeito" (existe
        exatamente um caminho entre quaisquer duas células)."""
        inicio = (0, 0)
        visitadas = {inicio}
        pilha = [inicio]
        while pilha:
            atual = pilha[-1]
            nao_visitadas = [v for v in self.vizinhos(atual) if v not in visitadas]
            if nao_visitadas:
                proxima = self.aleatorio.choice(nao_visitadas)
                self.arestas_abertas.add(frozenset((atual, proxima)))
                visitadas.add(proxima)
                pilha.append(proxima)
            else:
                pilha.pop()

    def _todas_arestas(self):
        arestas = set()
        for linha in range(self.tamanho):
            for coluna in range(self.tamanho):
                celula = (linha, coluna)
                for vizinho in self.vizinhos(celula):
                    arestas.add(frozenset((celula, vizinho)))
        return arestas

    def _adicionar_atalhos(self, fracao):
        """Abre uma fração extra de paredes fechadas, criando ciclos e
        caminhos alternativos (um labirinto perfeito só tem um caminho
        entre dois pontos; aqui adicionamos opções extras)."""
        if fracao <= 0:
            return
        fechadas = list(self._todas_arestas() - self.arestas_abertas)
        self.aleatorio.shuffle(fechadas)
        n_abrir = int(len(fechadas) * fracao)
        self.arestas_abertas.update(fechadas[:n_abrir])

    # ------------------------------------------------------------------
    # Busca em largura (BFS)
    # ------------------------------------------------------------------
    def _distancias_por_bfs(self, inicio):
        """Distância (em passos) do ponto `inicio` até cada célula alcançável."""
        distancias = {inicio: 0}
        fila = deque([inicio])
        while fila:
            atual = fila.popleft()
            for vizinho in self.vizinhos(atual):
                if vizinho not in distancias and self.passavel(atual, vizinho):
                    distancias[vizinho] = distancias[atual] + 1
                    fila.append(vizinho)
        return distancias

    def caminho_mais_curto(self, inicio, objetivo):
        """Caminho mais curto (BFS) entre duas células, ou None se não existir."""
        if inicio == objetivo:
            return [inicio]
        anterior = {inicio: None}
        fila = deque([inicio])
        while fila:
            atual = fila.popleft()
            if atual == objetivo:
                break
            for vizinho in self.vizinhos(atual):
                if vizinho not in anterior and self.passavel(atual, vizinho):
                    anterior[vizinho] = atual
                    fila.append(vizinho)
        if objetivo not in anterior:
            return None

        caminho = []
        celula = objetivo
        while celula is not None:
            caminho.append(celula)
            celula = anterior[celula]
        caminho.reverse()
        return caminho

    # ------------------------------------------------------------------
    # Seleção justa dos pontos de partida
    # ------------------------------------------------------------------
    def _celulas_da_borda(self):
        celulas = set()
        for i in range(self.tamanho):
            celulas.add((0, i))
            celulas.add((self.tamanho - 1, i))
            celulas.add((i, 0))
            celulas.add((i, self.tamanho - 1))
        celulas.discard(self.objetivo)
        return list(celulas)

    def _selecionar_pontos_justos(self, k):
        """Agrupa as células da borda por distância até o objetivo e
        escolhe `k` delas dentro do grupo mais distante que tenha opções
        suficientes (pontos difíceis, mas sempre igualmente distantes)."""
        grupos = defaultdict(list)
        for celula in self._celulas_da_borda():
            d = self.distancias.get(celula)
            if d is not None:
                grupos[d].append(celula)

        distancias_candidatas = sorted(
            (d for d, celulas in grupos.items() if len(celulas) >= k), reverse=True
        )
        distancia_escolhida = (
            distancias_candidatas[0] if distancias_candidatas
            else max(grupos, key=lambda d: len(grupos[d]))
        )

        grupo = grupos[distancia_escolhida]
        escolhidas = self._selecionar_espalhados(grupo, min(k, len(grupo)))
        return escolhidas, distancia_escolhida

    def _selecionar_espalhados(self, celulas, k):
        """Escolhe k células tentando maximizar a distância entre elas,
        para que os pontos de partida fiquem espalhados pelo labirinto."""
        if len(celulas) <= k:
            return list(celulas)
        escolhidas = [self.aleatorio.choice(celulas)]
        restantes = [c for c in celulas if c != escolhidas[0]]
        while len(escolhidas) < k and restantes:
            melhor = max(restantes, key=lambda c: min(self._manhattan(c, e) for e in escolhidas))
            escolhidas.append(melhor)
            restantes.remove(melhor)
        return escolhidas

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])