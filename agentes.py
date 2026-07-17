"""
agents.py
---------
Três agentes que competem no mesmo labirinto, cada um com uma estratégia
diferente para chegar ao objetivo. Todos compartilham a mesma interface
(AgenteBase), então o jogo pode tratá-los de forma intercambiável.

1. AgenteAEstrela   -> busca heurística A* (estado, objetivo, heurística)
2. AgenteQLearning  -> aprendizado por reforço (tabela Q treinada antes do jogo)
3. AgenteGenetico   -> algoritmo genético (evolui uma sequência de ações)
"""

import heapq
import random

from labirinto import MOVIMENTOS

N_ACOES = len(MOVIMENTOS)


def mover(labirinto, posicao, acao):
    """Aplica uma ação a partir de `posicao` e devolve a célula resultante.
    Se a ação esbarrar numa parede ou sair do grid, a posição não muda."""
    dl, dc = MOVIMENTOS[acao]
    candidata = (posicao[0] + dl, posicao[1] + dc)
    if labirinto.dentro_dos_limites(candidata) and labirinto.passavel(posicao, candidata):
        return candidata
    return posicao


class AgenteBase:
    def __init__(self, nome, cor, rotulo=None):
        self.nome = nome
        self.rotulo = rotulo or nome
        self.cor = cor
        self.posicao = None
        self.concluido = False
        self.passos_dados = 0
        self.indice_caminho = 0  # usado pelos agentes que seguem uma lista de células

    def preparar_rodada(self, labirinto, partida):
        """Chamado no início de cada rodada, com um novo ponto de partida."""
        self.posicao = partida
        self.partida = partida
        self.concluido = False
        self.passos_dados = 0
        self.indice_caminho = 0

    def passo(self, labirinto):
        """Executa um passo no labirinto. Cada subclasse define sua estratégia."""
        raise NotImplementedError

    def _seguir_celulas(self, labirinto, celulas):
        """Avança uma célula por vez numa lista já calculada (usado pelos
        agentes A* e Genético, que traçam o caminho inteiro de antemão)."""
        if self.concluido:
            return
        if self.indice_caminho < len(celulas) - 1:
            self.indice_caminho += 1
            self.posicao = celulas[self.indice_caminho]
            self.passos_dados += 1
        if self.posicao == labirinto.objetivo:
            self.concluido = True


# ======================================================================
# 1) AGENTE HEURÍSTICO -- Busca A*
# ======================================================================
class AgenteAEstrela(AgenteBase):
    """
    Estado: posição (linha, coluna). Objetivo: chegar em labirinto.objetivo.
    Heurística: distância de Manhattan (admissível, pois cada passo custa 1
    e nunca é possível "cortar caminho" além do que ela prevê).
    O caminho inteiro é calculado uma vez em preparar_rodada(); passo()
    apenas o percorre.
    """

    def __init__(self, cor):
        super().__init__("Busca Heurística (A*)", cor, rotulo="Heurística")
        self.caminho = []

    def preparar_rodada(self, labirinto, partida):
        super().preparar_rodada(labirinto, partida)
        self.caminho = self._busca_a_estrela(labirinto, partida, labirinto.objetivo)

    @staticmethod
    def _heuristica(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _busca_a_estrela(self, labirinto, inicio, objetivo):
        contador = 0  # desempate estável no heap (evita comparar posições)
        fila_prioridade = [(self._heuristica(inicio, objetivo), contador, inicio)]
        veio_de = {}
        custo_g = {inicio: 0}
        visitadas = set()

        while fila_prioridade:
            _, _, atual = heapq.heappop(fila_prioridade)
            if atual in visitadas:
                continue
            visitadas.add(atual)

            if atual == objetivo:
                return self._reconstruir_caminho(veio_de, atual)

            for vizinho in labirinto.vizinhos(atual):
                if not labirinto.passavel(atual, vizinho) or vizinho in visitadas:
                    continue
                novo_custo_g = custo_g[atual] + 1
                if novo_custo_g < custo_g.get(vizinho, float("inf")):
                    custo_g[vizinho] = novo_custo_g
                    veio_de[vizinho] = atual
                    contador += 1
                    heapq.heappush(fila_prioridade, (novo_custo_g + self._heuristica(vizinho, objetivo), contador, vizinho))

        return [inicio]  # objetivo inalcançável (não deve ocorrer em labirinto conectado)

    @staticmethod
    def _reconstruir_caminho(veio_de, atual):
        caminho = [atual]
        while atual in veio_de:
            atual = veio_de[atual]
            caminho.append(atual)
        caminho.reverse()
        return caminho

    def passo(self, labirinto):
        self._seguir_celulas(labirinto, self.caminho)


# ======================================================================
# 2) AGENTE DE APRENDIZADO POR REFORÇO -- Q-Learning
# ======================================================================
class AgenteQLearning(AgenteBase):
    """
    Estado: posição. Ação: {cima, baixo, esquerda, direita}.
    Recompensa: +100 no objetivo, -1 por passo, -5 ao esbarrar em parede.
    Atualização (Bellman): Q(s,a) += alpha * (r + gamma * max Q(s') - Q(s,a))

    O agente é TREINADO offline (treinar()) antes da 1ª rodada, a partir de
    todos os pontos de partida possíveis. Durante o jogo ele age de forma
    quase-gulosa (pequena chance de exploração), mostrando o que aprendeu.
    """

    def __init__(self, cor, taxa_aprendizagem=0.2, fator_desconto=0.9,
                 exploracao_treino=0.25, exploracao_jogo=0.05, episodios_treino=6000):
        super().__init__("Aprendizado por Reforço (Q-Learning)", cor, rotulo="Reforço")
        self.taxa_aprendizagem = taxa_aprendizagem
        self.fator_desconto = fator_desconto
        self.exploracao_treino = exploracao_treino
        self.exploracao_jogo = exploracao_jogo
        self.episodios_treino = episodios_treino 
        self.tabela_q = {}
        self.aleatorio = random.Random()
        self.treinado = False

    def _obter_q(self, estado):
        return self.tabela_q.setdefault(estado, [0.0] * N_ACOES)

    def _recompensa_do_movimento(self, labirinto, posicao, acao):
        """Aplica a ação e devolve (nova_posição, recompensa)."""
        nova_posicao = mover(labirinto, posicao, acao)
        if nova_posicao == posicao:
            return nova_posicao, -5.0  # bateu na parede: fica parado, penalidade
        return nova_posicao, (100.0 if nova_posicao == labirinto.objetivo else -1.0)

    def treinar(self, labirinto, pontos_partida):
        """Fase de treinamento offline (roda uma vez antes da 1ª rodada)."""
        max_passos = labirinto.tamanho * 8
        for _ in range(self.episodios_treino):
            posicao = self.aleatorio.choice(pontos_partida)
            passos = 0
            while posicao != labirinto.objetivo and passos < max_passos:
                q = self._obter_q(posicao)
                if self.aleatorio.random() < self.exploracao_treino:
                    acao = self.aleatorio.randrange(N_ACOES)
                else:
                    acao = max(range(N_ACOES), key=lambda a: q[a])

                nova_posicao, recompensa = self._recompensa_do_movimento(labirinto, posicao, acao)
                proximo_q = self._obter_q(nova_posicao)
                q[acao] += self.taxa_aprendizagem * (recompensa + self.fator_desconto * max(proximo_q) - q[acao])
                posicao = nova_posicao
                passos += 1
        self.treinado = True

    def preparar_rodada(self, labirinto, partida):
        super().preparar_rodada(labirinto, partida)
        self.melhor_distancia_vista = labirinto.distancias.get(partida, 10 ** 6)
        self.contador_estagnado = 0

    def passo(self, labirinto):
        if self.concluido:
            return

        # Como a rodada só termina quando TODOS os agentes chegam, o agente
        # não pode ficar preso oscilando entre duas células por causa de
        # valores de Q empatados. Se ficar muito tempo sem se aproximar do
        # objetivo, aumentamos temporariamente a exploração.
        exploracao_efetiva = self.exploracao_jogo if self.contador_estagnado <= 60 else 0.6

        q = self._obter_q(self.posicao)
        if self.aleatorio.random() < exploracao_efetiva:
            acao = self.aleatorio.randrange(N_ACOES)
        else:
            melhor_valor = max(q)
            melhores_acoes = [a for a in range(N_ACOES) if q[a] == melhor_valor]
            acao = self.aleatorio.choice(melhores_acoes)  # desempate aleatório evita ciclos

        self.posicao, _ = self._recompensa_do_movimento(labirinto, self.posicao, acao)
        self.passos_dados += 1

        distancia_atual = labirinto.distancias.get(self.posicao, self.melhor_distancia_vista)
        if distancia_atual < self.melhor_distancia_vista:
            self.melhor_distancia_vista = distancia_atual
            self.contador_estagnado = 0
        else:
            self.contador_estagnado += 1

        if self.posicao == labirinto.objetivo:
            self.concluido = True


# ======================================================================
# 3) AGENTE DE ALGORITMO GENÉTICO
# ======================================================================
class AgenteGenetico(AgenteBase):
    """
    Indivíduo: sequência fixa de ações (genes em 0..3).
    Aptidão: quanto mais perto do objetivo o indivíduo termina (distância
    BFS real) e quanto menos passos usar, maior a aptidão -- com um grande
    bônus por realmente alcançar o objetivo.

    Como o ponto de partida muda a cada rodada, a população EVOLUI DE NOVO
    no início de cada rodada (preparar_rodada), mostrando seleção,
    cruzamento e mutação "ao vivo". Se não convergir dentro do orçamento
    normal de gerações, tentamos de novo com cromossomos maiores; como
    última rede de segurança, completamos o trajeto final com o caminho
    mais curto (BFS) -- o que fica registrado em usou_resgate.
    """

    def __init__(self, cor, tamanho_populacao=1500, geracoes=100,
                 taxa_mutacao=0.06, fracao_elite=0.1):
        super().__init__("Algoritmo Genético", cor, rotulo="Genético")
        self.tamanho_populacao = tamanho_populacao
        self.geracoes = geracoes
        self.taxa_mutacao = taxa_mutacao
        self.fracao_elite = fracao_elite
        self.aleatorio = random.Random()
        self.melhores_celulas = []
        self.historico_aptidao = []  # para depuração/relatório
        self.usou_resgate = False    # True se precisou da rede de segurança

    def preparar_rodada(self, labirinto, partida):
        super().preparar_rodada(labirinto, partida)
        self.usou_resgate = False

        tamanho_cromossomo = labirinto.tamanho * 3
        geracoes = self.geracoes
        celulas, historico = None, []

        for _tentativa in range(5):
            melhor_cromossomo, historico = self._evoluir(labirinto, partida, tamanho_cromossomo, geracoes)
            celulas = self._simular(labirinto, partida, melhor_cromossomo)
            if celulas[-1] == labirinto.objetivo:
                break
            # Não convergiu: dá mais "material genético" e mais tempo.
            tamanho_cromossomo = int(tamanho_cromossomo * 1.6)
            geracoes = int(geracoes * 1.4)

        if celulas[-1] != labirinto.objetivo:
            resgate = labirinto.caminho_mais_curto(celulas[-1], labirinto.objetivo)
            if resgate:
                celulas = celulas + resgate[1:]
                self.usou_resgate = True

        self.historico_aptidao = historico
        self.melhores_celulas = celulas

    def _cromossomo_aleatorio(self, tamanho):
        return [self.aleatorio.randrange(N_ACOES) for _ in range(tamanho)]

    def _simular(self, labirinto, partida, cromossomo):
        """Executa um cromossomo passo a passo e devolve as células visitadas."""
        posicao = partida
        celulas = [posicao]
        for gene in cromossomo:
            posicao = mover(labirinto, posicao, gene)
            celulas.append(posicao)
            if posicao == labirinto.objetivo:
                break
        return celulas

    def _aptidao(self, labirinto, partida, cromossomo):
        celulas = self._simular(labirinto, partida, cromossomo)
        final = celulas[-1]
        passos_usados = len(celulas) - 1
        distancia_ao_objetivo = labirinto.distancias.get(final, labirinto.tamanho * 2)
        nota = -distancia_ao_objetivo * 10 - passos_usados * 0.05
        if final == labirinto.objetivo:
            nota += 2000 - passos_usados
        return nota

    def _torneio(self, ranqueados, k=5):
        """Seleção por torneio: sorteia k indivíduos e devolve o melhor deles.
        `ranqueados` já está ordenado do melhor para o pior fitness."""
        indices = sorted(self.aleatorio.sample(range(len(ranqueados)), min(k, len(ranqueados))))
        return ranqueados[indices[0]][0]

    def _cruzamento(self, pai1, pai2):
        if len(pai1) < 2:
            return pai1[:]
        ponto = self.aleatorio.randrange(1, len(pai1))
        return pai1[:ponto] + pai2[ponto:]

    def _mutacao(self, cromossomo):
        return [self.aleatorio.randrange(N_ACOES) if self.aleatorio.random() < self.taxa_mutacao else gene
                for gene in cromossomo]

    def _evoluir(self, labirinto, partida, tamanho_cromossomo, geracoes):
        populacao = [self._cromossomo_aleatorio(tamanho_cromossomo) for _ in range(self.tamanho_populacao)]
        melhor_geral, melhor_aptidao_geral = None, float("-inf")
        historico = []

        for _geracao in range(geracoes):
            ranqueados = sorted(
                ((c, self._aptidao(labirinto, partida, c)) for c in populacao),
                key=lambda par: par[1],
                reverse=True,
            )
            historico.append(ranqueados[0][1])
            if ranqueados[0][1] > melhor_aptidao_geral:
                melhor_aptidao_geral = ranqueados[0][1]
                melhor_geral = ranqueados[0][0]

            n_elite = max(2, int(self.tamanho_populacao * self.fracao_elite))
            proxima_geracao = [c for c, _ in ranqueados[:n_elite]]  # elitismo: os melhores passam direto

            while len(proxima_geracao) < self.tamanho_populacao:
                pai1 = self._torneio(ranqueados)
                pai2 = self._torneio(ranqueados)
                filho = self._mutacao(self._cruzamento(pai1, pai2))
                proxima_geracao.append(filho)

            populacao = proxima_geracao

        return melhor_geral, historico

    def passo(self, labirinto):
        self._seguir_celulas(labirinto, self.melhores_celulas)