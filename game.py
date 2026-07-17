"""
game.py
-------
Interface visual (pygame) da competição entre os três agentes.

- Spawns fixos, sorteados a cada rodada com distância justa: ver
  maze.py (Maze._select_fair_spawns) e _start_round() abaixo.
- Camada de sobreposição (2+ agentes na mesma célula): draw_agents_layer().
- Placar por rodada: _finish_round().
- Botões de reiniciar e pular rodada: _handle_click().
"""

import random
import sys
import pygame

from maze import Maze
from agents import AStarAgent, QLearningAgent, GeneticAgent


# ----------------------------------------------------------------------
# Configurações visuais e de jogo
# ----------------------------------------------------------------------
CELL_SIZE = 26
MAZE_SIZE = 17            # labirinto médio (17x17 células)
EXTRA_CONNECTIONS = 0.15  # fração de paredes extras abertas -> mais caminhos
N_SPAWNS = 4              # pontos fixos de spawn (>= número de agentes)
STEP_DELAY_MS = 140       # velocidade de animação (ms entre passos dos agentes)
TOTAL_ROUNDS = 3

SIDEBAR_WIDTH = 300
MARGIN = 20

COLOR_BG = (250, 248, 240)
COLOR_WALL = (40, 40, 40)
COLOR_MAZE_BG = (255, 255, 255)
COLOR_GOAL = (220, 60, 60)
COLOR_SPAWN_MARK = (120, 120, 200)
COLOR_OVERLAP_BG = (255, 210, 0)
COLOR_OVERLAP_BORDER = (170, 110, 0)
COLOR_TEXT = (30, 30, 30)
COLOR_BUTTON = (70, 110, 200)
COLOR_BUTTON_HOVER = (95, 135, 225)
COLOR_BUTTON_TEXT = (255, 255, 255)
COLOR_PANEL = (235, 235, 245)
COLOR_DIVIDER = (200, 200, 210)

AGENT_COLORS = {
    "astar": (220, 40, 40),   # vermelho
    "rl": (30, 100, 220),     # azul
    "ga": (30, 160, 60),      # verde
}

PLACEMENT_POINTS = [3, 2, 1]  # pontos para 1º, 2º, 3º lugar


class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def draw(self, screen, font, mouse_pos):
        color = COLOR_BUTTON_HOVER if self.rect.collidepoint(mouse_pos) else COLOR_BUTTON
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, (30, 30, 30), self.rect, width=2, border_radius=8)
        text = font.render(self.label, True, COLOR_BUTTON_TEXT)
        screen.blit(text, text.get_rect(center=self.rect.center))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class GameApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Competição de Agentes Inteligentes - Labirinto")

        maze_pixels = MAZE_SIZE * CELL_SIZE
        self.origin = (MARGIN, MARGIN + 40)
        width = maze_pixels + SIDEBAR_WIDTH + MARGIN * 3
        height = max(maze_pixels + MARGIN * 2 + 40, 560)
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("arial", 16)
        self.font_bold = pygame.font.SysFont("arial", 18, bold=True)
        self.font_title = pygame.font.SysFont("arial", 24, bold=True)

        self.rng = random.Random()
        self.total_rounds = TOTAL_ROUNDS

        self.agents = [
            AStarAgent(AGENT_COLORS["astar"]),
            QLearningAgent(AGENT_COLORS["rl"]),
            GeneticAgent(AGENT_COLORS["ga"]),
        ]

        btn_x = self.origin[0] + maze_pixels + MARGIN
        self.btn_restart = Button((btn_x, height - 120, SIDEBAR_WIDTH - MARGIN, 44), "Reiniciar Jogo")
        self.btn_next = Button((btn_x, height - 64, SIDEBAR_WIDTH - MARGIN, 44), "Pular / Próxima Rodada")

        self.status_message = ""
        self._full_restart()

    # ------------------------------------------------------------------
    # Ciclo de vida do jogo
    # ------------------------------------------------------------------
    def _full_restart(self):
        self.round_num = 0
        self.game_over = False
        self.scores = {a.name: 0 for a in self.agents}
        self.rounds_history = []
        self.maze = Maze(size=MAZE_SIZE, extra_connections=EXTRA_CONNECTIONS, n_spawns=N_SPAWNS)

        self.status_message = "Treinando o agente de aprendizado por reforço..."
        self._render_status_only()
        rl_agent = next(a for a in self.agents if isinstance(a, QLearningAgent))
        rl_agent.train(self.maze, self.maze.spawn_points)

        self._start_round()

    def _start_round(self):
        self.round_num += 1
        pool = list(self.maze.spawn_points)
        self.rng.shuffle(pool)
        assignments = pool[: len(self.agents)]

        self.status_message = "Calculando plano do agente genético..."
        self._render_status_only()

        for agent, spawn in zip(self.agents, assignments):
            agent.prepare_round(self.maze, spawn)

        self.round_finish_order = []
        self.round_active = True
        self.last_step_time = pygame.time.get_ticks()
        self.status_message = f"Rodada {self.round_num} de {self.total_rounds} em andamento..."

    def _finish_round(self):
        order = list(self.round_finish_order)
        remaining = [a for a in self.agents if a not in order]
        remaining.sort(key=lambda a: self.maze.distances.get(a.pos, 10 ** 6))
        full_order = order + remaining

        for i, agent in enumerate(full_order):
            points = PLACEMENT_POINTS[i] if i < len(PLACEMENT_POINTS) else 0
            self.scores[agent.name] += points

        self.rounds_history.append([(a.name, a.finished, a.steps_taken) for a in full_order])
        self.round_active = False

        if self.round_num >= self.total_rounds:
            self.game_over = True
            self.status_message = "Jogo concluído! Veja o placar final."
        else:
            names = " > ".join(a.short_label for a in full_order)
            self.status_message = f"Rodada {self.round_num} encerrada: {names}"

    def update(self):
        if not self.round_active:
            return
        now = pygame.time.get_ticks()
        if now - self.last_step_time < STEP_DELAY_MS:
            return
        self.last_step_time = now

        for agent in self.agents:
            if not agent.finished:
                agent.step(self.maze)
                if agent.finished and agent not in self.round_finish_order:
                    self.round_finish_order.append(agent)

        # A rodada só termina automaticamente quando TODOS os agentes
        # chegam ao objetivo (ou quando o usuário clica em "Pular").
        if all(a.finished for a in self.agents):
            self._finish_round()

    # ------------------------------------------------------------------
    # Entrada do usuário
    # ------------------------------------------------------------------
    def _handle_click(self, pos):
        if self.btn_restart.is_clicked(pos):
            self._full_restart()
            return
        if self.btn_next.is_clicked(pos):
            if self.game_over:
                return
            if self.round_active:
                self._finish_round()
            if not self.game_over:
                self._start_round()

    # ------------------------------------------------------------------
    # Desenho
    # ------------------------------------------------------------------
    def draw_maze(self):
        ox, oy = self.origin
        size_px = self.maze.size * CELL_SIZE
        pygame.draw.rect(self.screen, COLOR_MAZE_BG, (ox, oy, size_px, size_px))

        for r in range(self.maze.size):
            for c in range(self.maze.size):
                cell = (r, c)
                x, y = ox + c * CELL_SIZE, oy + r * CELL_SIZE
                right, down = (r, c + 1), (r + 1, c)
                if not self.maze.in_bounds(right) or not self.maze.passable(cell, right):
                    pygame.draw.line(self.screen, COLOR_WALL, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), 3)
                if not self.maze.in_bounds(down) or not self.maze.passable(cell, down):
                    pygame.draw.line(self.screen, COLOR_WALL, (x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), 3)

        pygame.draw.line(self.screen, COLOR_WALL, (ox, oy), (ox, oy + size_px), 3)
        pygame.draw.line(self.screen, COLOR_WALL, (ox, oy), (ox + size_px, oy), 3)

        # pontos fixos de spawn (referência visual, mesmo os não usados na rodada)
        for sr, sc in self.maze.spawn_points:
            cx, cy = ox + sc * CELL_SIZE + CELL_SIZE // 2, oy + sr * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(self.screen, COLOR_SPAWN_MARK, (cx, cy), 4)

        gr, gc = self.maze.goal
        pygame.draw.rect(
            self.screen, COLOR_GOAL,
            (ox + gc * CELL_SIZE + 4, oy + gr * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8),
            border_radius=4,
        )

    def draw_agents_layer(self):
        """Destaca em amarelo qualquer célula onde 2+ agentes estejam juntos,
        desenhando cada um em um quadrante da célula para todos ficarem visíveis."""
        ox, oy = self.origin
        groups = {}
        for agent in self.agents:
            groups.setdefault(agent.pos, []).append(agent)

        offsets_2x2 = [(0, 0), (1, 0), (0, 1), (1, 1)]

        for (r, c), group in groups.items():
            x, y = ox + c * CELL_SIZE, oy + r * CELL_SIZE

            if len(group) > 1:
                pygame.draw.rect(self.screen, COLOR_OVERLAP_BG, (x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2))
                pygame.draw.rect(self.screen, COLOR_OVERLAP_BORDER, (x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2), 2)

            if len(group) == 1:
                cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2
                radius = max(CELL_SIZE // 2 - 4, 3)
                pygame.draw.circle(self.screen, group[0].color, (cx, cy), radius)
                pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), radius, 1)
            else:
                sub = CELL_SIZE // 2
                for i, agent in enumerate(group):
                    dx, dy = offsets_2x2[i % 4]
                    cx, cy = x + dx * sub + sub // 2, y + dy * sub + sub // 2
                    radius = max(sub // 2 - 1, 2)
                    pygame.draw.circle(self.screen, agent.color, (cx, cy), radius)
                    pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), radius, 1)

    def _draw_divider(self, panel_x, panel_w, y):
        pygame.draw.line(self.screen, COLOR_DIVIDER, (panel_x + 14, y), (panel_x + panel_w - 14, y), 1)

    def draw_sidebar(self):
        ox, oy = self.origin
        maze_px = self.maze.size * CELL_SIZE
        panel_x = ox + maze_px + MARGIN
        panel_y = oy
        panel_w = SIDEBAR_WIDTH - MARGIN
        panel_h = self.maze.size * CELL_SIZE

        pygame.draw.rect(self.screen, COLOR_PANEL, (panel_x, panel_y, panel_w, panel_h), border_radius=10)

        y = panel_y + 14
        title = self.font_bold.render(f"Rodada {self.round_num} / {self.total_rounds}", True, COLOR_TEXT)
        self.screen.blit(title, (panel_x + 14, y))
        y += 36
        self._draw_divider(panel_x, panel_w, y)
        y += 14

        header = self.font_bold.render("Placar (acumulado)", True, COLOR_TEXT)
        self.screen.blit(header, (panel_x + 14, y))
        y += 26

        ranking = sorted(self.agents, key=lambda a: self.scores[a.name], reverse=True)
        for agent in ranking:
            pygame.draw.circle(self.screen, agent.color, (panel_x + 22, y + 9), 8)
            pygame.draw.circle(self.screen, (0, 0, 0), (panel_x + 22, y + 9), 8, 1)
            text = self.font.render(f"{agent.short_label}: {self.scores[agent.name]} pts", True, COLOR_TEXT)
            self.screen.blit(text, (panel_x + 38, y))
            y += 24

        y += 10
        self._draw_divider(panel_x, panel_w, y)
        y += 14

        header2 = self.font_bold.render("Status da rodada", True, COLOR_TEXT)
        self.screen.blit(header2, (panel_x + 14, y))
        y += 24

        for agent in self.agents:
            state = "chegou!" if agent.finished else "em andamento"
            text = self.font.render(f"{agent.short_label}: {agent.steps_taken} passos ({state})", True, COLOR_TEXT)
            self.screen.blit(text, (panel_x + 14, y))
            y += 20

        y += 14
        self._draw_divider(panel_x, panel_w, y)
        y += 10

        legend_lines = [
            "Legenda:",
            "• Ponto vermelho: objetivo",
            "• Pontos azuis pequenos: spawns fixos",
            "• Célula amarela: mais de 1 agente ali",
        ]
        for line in legend_lines:
            text = self.font.render(line, True, COLOR_TEXT)
            self.screen.blit(text, (panel_x + 14, y))
            y += 18

    def draw_top_status(self):
        text = self.font.render(self.status_message, True, COLOR_TEXT)
        self.screen.blit(text, (self.origin[0], 12))

    def draw_final_screen_overlay(self):
        if not self.game_over:
            return
        ox, oy = self.origin
        maze_px = self.maze.size * CELL_SIZE
        overlay = pygame.Surface((maze_px, maze_px), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 210))
        self.screen.blit(overlay, (ox, oy))

        ranking = sorted(self.agents, key=lambda a: self.scores[a.name], reverse=True)
        title = self.font_title.render("Resultado Final", True, COLOR_TEXT)
        self.screen.blit(title, (ox + maze_px // 2 - title.get_width() // 2, oy + 30))

        y = oy + 90
        for i, agent in enumerate(ranking):
            text = self.font_bold.render(
                f"{i + 1}º lugar: {agent.short_label} — {self.scores[agent.name]} pts", True, agent.color
            )
            self.screen.blit(text, (ox + maze_px // 2 - text.get_width() // 2, y))
            y += 34

    def draw(self):
        self.screen.fill(COLOR_BG)
        self.draw_top_status()
        self.draw_maze()
        self.draw_agents_layer()
        self.draw_final_screen_overlay()
        self.draw_sidebar()

        mouse_pos = pygame.mouse.get_pos()
        self.btn_restart.draw(self.screen, self.font, mouse_pos)
        self.btn_next.draw(self.screen, self.font, mouse_pos)

        pygame.display.flip()

    def _render_status_only(self):
        """Mostra uma mensagem de 'carregando' enquanto o treino do RL ou a
        evolução do GA acontecem, evitando a sensação de tela travada."""
        self.screen.fill(COLOR_BG)
        self.draw_top_status()
        pygame.display.flip()
        pygame.event.pump()

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    GameApp().run()