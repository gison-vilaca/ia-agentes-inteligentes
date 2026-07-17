"""
game.py
-------
Interface visual (pygame) da competição entre os três agentes.

- Pontos de partida fixos, sorteados a cada rodada com distância justa:
  ver maze.py (Labirinto._selecionar_pontos_justos) e _iniciar_rodada() abaixo.
- Camada de sobreposição (2+ agentes na mesma célula): desenhar_camada_agentes().
- Placar por rodada: _encerrar_rodada().
- Botões de reiniciar e pular rodada: _tratar_clique().
"""

import random
import sys
import pygame

from labirinto import Labirinto
from agentes import AgenteAEstrela, AgenteQLearning, AgenteGenetico


# ----------------------------------------------------------------------
# Configurações visuais e de jogo
# ----------------------------------------------------------------------
TAMANHO_CELULA = 26
TAMANHO_LABIRINTO = 17     # labirinto médio (17x17 células)
CONEXOES_EXTRAS = 0.15     # fração de paredes extras abertas -> mais caminhos
N_PONTOS_PARTIDA = 4       # pontos fixos de partida (>= número de agentes)
ATRASO_PASSO_MS = 140      # velocidade de animação (ms entre passos dos agentes)
TOTAL_RODADAS = 10

LARGURA_PAINEL = 300
MARGEM = 20

COR_FUNDO = (250, 248, 240)
COR_PAREDE = (40, 40, 40)
COR_FUNDO_LABIRINTO = (255, 255, 255)
COR_OBJETIVO = (220, 60, 60)
COR_MARCA_PARTIDA = (120, 120, 200)
COR_SOBREPOSICAO_FUNDO = (255, 210, 0)
COR_SOBREPOSICAO_BORDA = (170, 110, 0)
COR_TEXTO = (30, 30, 30)
COR_BOTAO = (70, 110, 200)
COR_BOTAO_HOVER = (95, 135, 225)
COR_TEXTO_BOTAO = (255, 255, 255)
COR_PAINEL = (235, 235, 245)
COR_DIVISOR = (200, 200, 210)

CORES_AGENTES = {
    "astar": (220, 40, 40),  # vermelho
    "rl": (30, 100, 220),    # azul
    "ga": (30, 160, 60),     # verde
}

PONTOS_COLOCACAO = [3, 2, 1]  # pontos para 1º, 2º, 3º lugar


class Botao:
    def __init__(self, retangulo, rotulo):
        self.retangulo = pygame.Rect(retangulo)
        self.rotulo = rotulo

    def desenhar(self, tela, fonte, posicao_mouse):
        cor = COR_BOTAO_HOVER if self.retangulo.collidepoint(posicao_mouse) else COR_BOTAO
        pygame.draw.rect(tela, cor, self.retangulo, border_radius=8)
        pygame.draw.rect(tela, (30, 30, 30), self.retangulo, width=2, border_radius=8)
        texto = fonte.render(self.rotulo, True, COR_TEXTO_BOTAO)
        tela.blit(texto, texto.get_rect(center=self.retangulo.center))

    def foi_clicado(self, posicao):
        return self.retangulo.collidepoint(posicao)


class Jogo:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Competição de Agentes Inteligentes - Labirinto")

        pixels_labirinto = TAMANHO_LABIRINTO * TAMANHO_CELULA
        self.origem = (MARGEM, MARGEM + 40)
        largura = pixels_labirinto + LARGURA_PAINEL + MARGEM * 3
        altura = max(pixels_labirinto + MARGEM * 2 + 40, 560)
        self.tela = pygame.display.set_mode((largura, altura))
        self.relogio = pygame.time.Clock()

        self.fonte = pygame.font.SysFont("arial", 16)
        self.fonte_negrito = pygame.font.SysFont("arial", 18, bold=True)
        self.fonte_titulo = pygame.font.SysFont("arial", 24, bold=True)

        self.aleatorio = random.Random()
        self.total_rodadas = TOTAL_RODADAS

        self.agentes = [
            AgenteAEstrela(CORES_AGENTES["astar"]),
            AgenteQLearning(CORES_AGENTES["rl"]),
            AgenteGenetico(CORES_AGENTES["ga"]),
        ]

        x_botoes = self.origem[0] + pixels_labirinto + MARGEM
        self.botao_reiniciar = Botao((x_botoes, altura - 120, LARGURA_PAINEL - MARGEM, 44), "Reiniciar Jogo")
        self.botao_proxima = Botao((x_botoes, altura - 64, LARGURA_PAINEL - MARGEM, 44), "Pular / Próxima Rodada")

        self.mensagem_status = ""
        self._reiniciar_tudo()

    # ------------------------------------------------------------------
    # Ciclo de vida do jogo
    # ------------------------------------------------------------------
    def _reiniciar_tudo(self):
        self.numero_rodada = 0
        self.jogo_encerrado = False
        self.placar = {a.nome: 0 for a in self.agentes}
        self.historico_rodadas = []
        self.labirinto = Labirinto(tamanho=TAMANHO_LABIRINTO, conexoes_extras=CONEXOES_EXTRAS, n_pontos_partida=N_PONTOS_PARTIDA)

        self.mensagem_status = "Treinando o agente de aprendizado por reforço..."
        self._mostrar_status()
        agente_rl = next(a for a in self.agentes if isinstance(a, AgenteQLearning))
        agente_rl.treinar(self.labirinto, self.labirinto.pontos_partida)

        self._iniciar_rodada()

    def _iniciar_rodada(self):
        self.numero_rodada += 1
        disponiveis = list(self.labirinto.pontos_partida)
        self.aleatorio.shuffle(disponiveis)
        sorteados = disponiveis[: len(self.agentes)]

        self.mensagem_status = "Calculando plano do agente genético..."
        self._mostrar_status()

        for agente, partida in zip(self.agentes, sorteados):
            agente.preparar_rodada(self.labirinto, partida)

        self.ordem_chegada = []
        self.rodada_ativa = True
        self.tempo_ultimo_passo = pygame.time.get_ticks()
        self.mensagem_status = f"Rodada {self.numero_rodada} de {self.total_rodadas} em andamento..."

    def _encerrar_rodada(self):
        ordem = list(self.ordem_chegada)
        restantes = [a for a in self.agentes if a not in ordem]
        restantes.sort(key=lambda a: self.labirinto.distancias.get(a.posicao, 10 ** 6))
        ordem_final = ordem + restantes

        for i, agente in enumerate(ordem_final):
            pontos = PONTOS_COLOCACAO[i] if i < len(PONTOS_COLOCACAO) else 0
            self.placar[agente.nome] += pontos

        self.historico_rodadas.append([(a.nome, a.concluido, a.passos_dados) for a in ordem_final])
        self.rodada_ativa = False

        if self.numero_rodada >= self.total_rodadas:
            self.jogo_encerrado = True
            self.mensagem_status = "Jogo concluído! Veja o placar final."
        else:
            nomes = " > ".join(a.rotulo for a in ordem_final)
            self.mensagem_status = f"Rodada {self.numero_rodada} encerrada: {nomes}"

    def atualizar(self):
        if not self.rodada_ativa:
            return
        agora = pygame.time.get_ticks()
        if agora - self.tempo_ultimo_passo < ATRASO_PASSO_MS:
            return
        self.tempo_ultimo_passo = agora

        for agente in self.agentes:
            if not agente.concluido:
                agente.passo(self.labirinto)
                if agente.concluido and agente not in self.ordem_chegada:
                    self.ordem_chegada.append(agente)

        # A rodada só termina automaticamente quando TODOS os agentes
        # chegam ao objetivo (ou quando o usuário clica em "Pular").
        if all(a.concluido for a in self.agentes):
            self._encerrar_rodada()

    # ------------------------------------------------------------------
    # Entrada do usuário
    # ------------------------------------------------------------------
    def _tratar_clique(self, posicao):
        if self.botao_reiniciar.foi_clicado(posicao):
            self._reiniciar_tudo()
            return
        if self.botao_proxima.foi_clicado(posicao):
            if self.jogo_encerrado:
                return
            if self.rodada_ativa:
                self._encerrar_rodada()
            if not self.jogo_encerrado:
                self._iniciar_rodada()

    # ------------------------------------------------------------------
    # Desenho
    # ------------------------------------------------------------------
    def desenhar_labirinto(self):
        ox, oy = self.origem
        tamanho_px = self.labirinto.tamanho * TAMANHO_CELULA
        pygame.draw.rect(self.tela, COR_FUNDO_LABIRINTO, (ox, oy, tamanho_px, tamanho_px))

        for linha in range(self.labirinto.tamanho):
            for coluna in range(self.labirinto.tamanho):
                celula = (linha, coluna)
                x, y = ox + coluna * TAMANHO_CELULA, oy + linha * TAMANHO_CELULA
                direita, abaixo = (linha, coluna + 1), (linha + 1, coluna)
                if not self.labirinto.dentro_dos_limites(direita) or not self.labirinto.passavel(celula, direita):
                    pygame.draw.line(self.tela, COR_PAREDE, (x + TAMANHO_CELULA, y), (x + TAMANHO_CELULA, y + TAMANHO_CELULA), 3)
                if not self.labirinto.dentro_dos_limites(abaixo) or not self.labirinto.passavel(celula, abaixo):
                    pygame.draw.line(self.tela, COR_PAREDE, (x, y + TAMANHO_CELULA), (x + TAMANHO_CELULA, y + TAMANHO_CELULA), 3)

        pygame.draw.line(self.tela, COR_PAREDE, (ox, oy), (ox, oy + tamanho_px), 3)
        pygame.draw.line(self.tela, COR_PAREDE, (ox, oy), (ox + tamanho_px, oy), 3)

        # pontos fixos de partida (referência visual, mesmo os não usados na rodada)
        for lin, col in self.labirinto.pontos_partida:
            cx, cy = ox + col * TAMANHO_CELULA + TAMANHO_CELULA // 2, oy + lin * TAMANHO_CELULA + TAMANHO_CELULA // 2
            pygame.draw.circle(self.tela, COR_MARCA_PARTIDA, (cx, cy), 4)

        lg, cg = self.labirinto.objetivo
        pygame.draw.rect(
            self.tela, COR_OBJETIVO,
            (ox + cg * TAMANHO_CELULA + 4, oy + lg * TAMANHO_CELULA + 4, TAMANHO_CELULA - 8, TAMANHO_CELULA - 8),
            border_radius=4,
        )

    def desenhar_camada_agentes(self):
        """Destaca em amarelo qualquer célula onde 2+ agentes estejam juntos,
        desenhando cada um em um quadrante da célula para todos ficarem visíveis."""
        ox, oy = self.origem
        grupos = {}
        for agente in self.agentes:
            grupos.setdefault(agente.posicao, []).append(agente)

        deslocamentos_2x2 = [(0, 0), (1, 0), (0, 1), (1, 1)]

        for (linha, coluna), grupo in grupos.items():
            x, y = ox + coluna * TAMANHO_CELULA, oy + linha * TAMANHO_CELULA

            if len(grupo) > 1:
                pygame.draw.rect(self.tela, COR_SOBREPOSICAO_FUNDO, (x + 1, y + 1, TAMANHO_CELULA - 2, TAMANHO_CELULA - 2))
                pygame.draw.rect(self.tela, COR_SOBREPOSICAO_BORDA, (x + 1, y + 1, TAMANHO_CELULA - 2, TAMANHO_CELULA - 2), 2)

            if len(grupo) == 1:
                cx, cy = x + TAMANHO_CELULA // 2, y + TAMANHO_CELULA // 2
                raio = max(TAMANHO_CELULA // 2 - 4, 3)
                pygame.draw.circle(self.tela, grupo[0].cor, (cx, cy), raio)
                pygame.draw.circle(self.tela, (0, 0, 0), (cx, cy), raio, 1)
            else:
                sub = TAMANHO_CELULA // 2
                for i, agente in enumerate(grupo):
                    dx, dy = deslocamentos_2x2[i % 4]
                    cx, cy = x + dx * sub + sub // 2, y + dy * sub + sub // 2
                    raio = max(sub // 2 - 1, 2)
                    pygame.draw.circle(self.tela, agente.cor, (cx, cy), raio)
                    pygame.draw.circle(self.tela, (0, 0, 0), (cx, cy), raio, 1)

    def _desenhar_divisor(self, x_painel, largura_painel, y):
        pygame.draw.line(self.tela, COR_DIVISOR, (x_painel + 14, y), (x_painel + largura_painel - 14, y), 1)

    def desenhar_painel_lateral(self):
        ox, oy = self.origem
        px_labirinto = self.labirinto.tamanho * TAMANHO_CELULA
        x_painel = ox + px_labirinto + MARGEM
        y_painel = oy
        largura_painel = LARGURA_PAINEL - MARGEM
        altura_painel = self.labirinto.tamanho * TAMANHO_CELULA

        pygame.draw.rect(self.tela, COR_PAINEL, (x_painel, y_painel, largura_painel, altura_painel), border_radius=10)

        y = y_painel + 14
        titulo = self.fonte_negrito.render(f"Rodada {self.numero_rodada} / {self.total_rodadas}", True, COR_TEXTO)
        self.tela.blit(titulo, (x_painel + 14, y))
        y += 36
        self._desenhar_divisor(x_painel, largura_painel, y)
        y += 14

        cabecalho = self.fonte_negrito.render("Placar (acumulado)", True, COR_TEXTO)
        self.tela.blit(cabecalho, (x_painel + 14, y))
        y += 26

        classificacao = sorted(self.agentes, key=lambda a: self.placar[a.nome], reverse=True)
        for agente in classificacao:
            pygame.draw.circle(self.tela, agente.cor, (x_painel + 22, y + 9), 8)
            pygame.draw.circle(self.tela, (0, 0, 0), (x_painel + 22, y + 9), 8, 1)
            texto = self.fonte.render(f"{agente.rotulo}: {self.placar[agente.nome]} pts", True, COR_TEXTO)
            self.tela.blit(texto, (x_painel + 38, y))
            y += 24

        y += 10
        self._desenhar_divisor(x_painel, largura_painel, y)
        y += 14

        cabecalho2 = self.fonte_negrito.render("Status da rodada", True, COR_TEXTO)
        self.tela.blit(cabecalho2, (x_painel + 14, y))
        y += 24

        for agente in self.agentes:
            estado = "chegou!" if agente.concluido else "em andamento"
            texto = self.fonte.render(f"{agente.rotulo}: {agente.passos_dados} passos ({estado})", True, COR_TEXTO)
            self.tela.blit(texto, (x_painel + 14, y))
            y += 20

        y += 14
        self._desenhar_divisor(x_painel, largura_painel, y)
        y += 10

        legenda = [
            "Legenda:",
            "• Ponto vermelho: objetivo",
            "• Pontos azuis pequenos: pontos de partida",
            "• Célula amarela: mais de 1 agente ali",
        ]
        for linha in legenda:
            texto = self.fonte.render(linha, True, COR_TEXTO)
            self.tela.blit(texto, (x_painel + 14, y))
            y += 18

    def desenhar_status_superior(self):
        texto = self.fonte.render(self.mensagem_status, True, COR_TEXTO)
        self.tela.blit(texto, (self.origem[0], 12))

    def desenhar_tela_final(self):
        if not self.jogo_encerrado:
            return
        ox, oy = self.origem
        px_labirinto = self.labirinto.tamanho * TAMANHO_CELULA
        sobreposicao = pygame.Surface((px_labirinto, px_labirinto), pygame.SRCALPHA)
        sobreposicao.fill((255, 255, 255, 210))
        self.tela.blit(sobreposicao, (ox, oy))

        classificacao = sorted(self.agentes, key=lambda a: self.placar[a.nome], reverse=True)
        titulo = self.fonte_titulo.render("Resultado Final", True, COR_TEXTO)
        self.tela.blit(titulo, (ox + px_labirinto // 2 - titulo.get_width() // 2, oy + 30))

        y = oy + 90
        for i, agente in enumerate(classificacao):
            texto = self.fonte_negrito.render(
                f"{i + 1}º lugar: {agente.rotulo} — {self.placar[agente.nome]} pts", True, agente.cor
            )
            self.tela.blit(texto, (ox + px_labirinto // 2 - texto.get_width() // 2, y))
            y += 34

    def desenhar(self):
        self.tela.fill(COR_FUNDO)
        self.desenhar_status_superior()
        self.desenhar_labirinto()
        self.desenhar_camada_agentes()
        self.desenhar_tela_final()
        self.desenhar_painel_lateral()

        posicao_mouse = pygame.mouse.get_pos()
        self.botao_reiniciar.desenhar(self.tela, self.fonte, posicao_mouse)
        self.botao_proxima.desenhar(self.tela, self.fonte, posicao_mouse)

        pygame.display.flip()

    def _mostrar_status(self):
        """Mostra uma mensagem de 'carregando' enquanto o treino do RL ou a
        evolução do GA acontecem, evitando a sensação de tela travada."""
        self.tela.fill(COR_FUNDO)
        self.desenhar_status_superior()
        pygame.display.flip()
        pygame.event.pump()

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------
    def executar(self):
        rodando = True
        while rodando:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    rodando = False
                elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                    self._tratar_clique(evento.pos)

            self.atualizar()
            self.desenhar()
            self.relogio.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Jogo().executar()