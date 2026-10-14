"""
game.py
-------
Modern, professional Pygame UI for the Selfie Puzzle Game.

Panel layout (top -> bottom):
  Title -> divider -> Time card -> Moves card -> Progress bar
  -> [Preview] [Restart] buttons -> divider -> Instructions

Extras:
  - Smooth fade-in on launch (no black flash / freeze feeling)
  - Animated smooth progress bar
  - Glowing correct-piece borders with pulse
  - Confetti victory screen
  - Smooth button hover transitions
"""

import pygame
import numpy as np
import cv2
import time
import math
from typing import Optional, Tuple
from puzzle import Puzzle, GRID_SIZE, NUM_PIECES

# ── Layout ────────────────────────────────────────────────────────────────────
PIECE_SIZE  = 160
GRID_PIXEL  = PIECE_SIZE * GRID_SIZE          # 480
PANEL_WIDTH = 300
BOARD_PAD   = 30
WINDOW_W    = GRID_PIXEL + PANEL_WIDTH + BOARD_PAD * 3
WINDOW_H    = GRID_PIXEL + BOARD_PAD * 2
BOARD_X     = BOARD_PAD
BOARD_Y     = BOARD_PAD
TILE_GAP    = 4
FPS         = 60
BR          = 10   # default border radius

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG             = (12,  14,  22)
C_PANEL          = (20,  24,  38)
C_PANEL_BORDER   = (38,  45,  70)
C_CARD           = (28,  33,  52)
C_CARD_BORDER    = (48,  57,  88)
C_BORDER         = (45,  54,  82)
C_BORDER_SOLVED  = (45, 210, 110)
C_BORDER_DRAG    = (90, 160, 255)
C_TEXT_PRIMARY   = (235, 238, 255)
C_TEXT_SECONDARY = (120, 130, 160)
C_TEXT_LABEL     = (90,  100, 135)
C_ACCENT         = (90,  150, 255)
C_GREEN          = (50,  200, 105)
C_GREEN_DIM      = (30,  120,  65)
C_BTN_PRIMARY    = (50,  100, 220)
C_BTN_PRIMARY_H  = (70,  125, 255)
C_BTN_SECONDARY  = (32,   38,  60)
C_BTN_SECONDARY_H= (45,   55,  88)
C_OVERLAY        = (8,    10,  18)
C_PROGRESS_BG    = (28,   33,  52)
C_PROGRESS_FILL  = (50,  200, 105)


# ── Helpers ───────────────────────────────────────────────────────────────────
def cv2_to_pygame(img: np.ndarray) -> pygame.Surface:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))


def draw_rounded_rect(surf, color, rect, radius=BR, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ── Button ────────────────────────────────────────────────────────────────────
class Button:
    def __init__(self, x, y, w, h, label, font,
                 color_normal=C_BTN_SECONDARY,
                 color_hover=C_BTN_SECONDARY_H,
                 border_color=C_CARD_BORDER,
                 text_color=C_TEXT_PRIMARY):
        self.rect          = pygame.Rect(x, y, w, h)
        self.label         = label
        self.font          = font
        self.color_normal  = color_normal
        self.color_hover   = color_hover
        self.border_color  = border_color
        self.text_color    = text_color
        self._hover_t      = 0.0

    def update(self, mouse_pos, dt=1/60):
        target = 1.0 if self.rect.collidepoint(mouse_pos) else 0.0
        self._hover_t += (target - self._hover_t) * min(dt * 12, 1.0)

    def draw(self, surface):
        bg = lerp_color(self.color_normal, self.color_hover, self._hover_t)
        draw_rounded_rect(surface, bg, self.rect, BR, 1, self.border_color)
        text_surf = self.font.render(self.label, True, self.text_color)
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center))

    def is_clicked(self, event) -> bool:
        return (event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.rect.collidepoint(event.pos))


# ── Main UI ───────────────────────────────────────────────────────────────────
class GameUI:
    def __init__(self, puzzle: Puzzle):
        pygame.init()
        pygame.display.set_caption("Selfie Puzzle Game")

        self.puzzle          = puzzle
        self.screen          = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock           = pygame.time.Clock()

        self.start_time      = time.time()
        self.move_count      = 0
        self.is_solved       = False
        self.show_preview    = False
        self.completion_time: Optional[float] = None

        self.dragging_pos:   Optional[int]            = None
        self.drag_surface:   Optional[pygame.Surface] = None
        self.drag_offset:    Tuple[int, int]          = (0, 0)
        self.mouse_pos:      Tuple[int, int]          = (0, 0)

        self._pulse         = 0.0
        self._victory_alpha = 0
        self._fade_alpha    = 255   # starts opaque, fades to 0 -> smooth launch
        self._confetti: list = []

        self._init_fonts()
        self.piece_surfaces = self._precompute_piece_surfaces()

        preview_img = puzzle.get_full_preview_image()
        self.preview_surface = (
            pygame.transform.scale(cv2_to_pygame(preview_img), (GRID_PIXEL, GRID_PIXEL))
            if preview_img is not None else None
        )

        self._build_panel_layout()
        self._init_confetti()
        self._progress_t = self._get_progress()

    # ── Fonts ─────────────────────────────────────────────────────────────────
    def _init_fonts(self):
        def sf(name, size, bold=False):
            try:    return pygame.font.SysFont(name, size, bold=bold)
            except: return pygame.font.Font(None, size + 6)

        self.f_title  = sf("segoeui", 30, bold=True)
        self.f_label  = sf("segoeui", 13)
        self.f_value  = sf("consolas", 22, bold=True)
        self.f_btn    = sf("segoeui", 16, bold=True)
        self.f_small  = sf("segoeui", 13)
        self.f_huge   = sf("segoeui", 50, bold=True)
        self.f_medium = sf("segoeui", 20, bold=True)

    # ── Panel layout ──────────────────────────────────────────────────────────
    def _build_panel_layout(self):
        px     = BOARD_X + GRID_PIXEL + BOARD_PAD
        pw     = PANEL_WIDTH
        cx     = px + pw // 2
        pad    = 16
        card_w = pw - pad * 2
        card_x = px + pad

        y = BOARD_PAD + 18

        # Title
        self._title_pos = (cx, y + 14)
        y += 54

        # Divider 1
        self._divider1 = ((px + 20, y), (px + pw - 20, y))
        y += 18

        # Time card
        self._card_time = pygame.Rect(card_x, y, card_w, 58)
        y += 68

        # Moves card
        self._card_moves = pygame.Rect(card_x, y, card_w, 58)
        y += 74

        # Progress bar label + bar
        self._prog_label_pos = (card_x, y)
        y += 22
        self._prog_rect = pygame.Rect(card_x, y, card_w, 10)
        y += 30

        # Buttons side by side
        btn_h   = 44
        btn_gap = 10
        btn_w   = (card_w - btn_gap) // 2
        self.btn_preview = Button(
            card_x, y, btn_w, btn_h, "Preview", self.f_btn,
            C_BTN_SECONDARY, C_BTN_SECONDARY_H, C_CARD_BORDER
        )
        self.btn_restart = Button(
            card_x + btn_w + btn_gap, y, btn_w, btn_h, "Restart", self.f_btn,
            C_BTN_PRIMARY, C_BTN_PRIMARY_H, C_BTN_PRIMARY_H
        )
        y += btn_h + 18

        # Divider 2
        self._divider2 = ((px + 20, y), (px + pw - 20, y))
        y += 16

        # Instructions start y
        self._instr_y  = y
        self._instr_cx = cx

    # ── Piece surfaces ────────────────────────────────────────────────────────
    def _precompute_piece_surfaces(self):
        surfaces = [None] * NUM_PIECES
        for piece in self.puzzle.pieces:
            img = pygame.transform.scale(
                cv2_to_pygame(piece.image), (PIECE_SIZE, PIECE_SIZE))
            surfaces[piece.correct_index] = img
        return surfaces

    # ── Confetti ──────────────────────────────────────────────────────────────
    def _init_confetti(self, n=100):
        rng = np.random
        self._confetti = [
            {
                "x":  float(rng.randint(0, WINDOW_W)),
                "y":  float(rng.uniform(-WINDOW_H, 0)),
                "vx": float(rng.uniform(-1.2, 1.2)),
                "vy": float(rng.uniform(2.5, 6.0)),
                "size": int(rng.randint(5, 12)),
                "color": (int(rng.randint(120,255)),
                          int(rng.randint(120,255)),
                          int(rng.randint(120,255))),
                "rot": float(rng.uniform(0, 360)),
                "rs":  float(rng.uniform(-4, 4)),
            } for _ in range(n)
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _tile_rect(self, position: int) -> pygame.Rect:
        row, col = self.puzzle.position_to_grid(position)
        x = BOARD_X + col * (PIECE_SIZE + TILE_GAP)
        y = BOARD_Y + row * (PIECE_SIZE + TILE_GAP)
        return pygame.Rect(x, y, PIECE_SIZE, PIECE_SIZE)

    def _pos_from_mouse(self, mx, my) -> Optional[int]:
        right  = BOARD_X + GRID_PIXEL + (GRID_SIZE - 1) * TILE_GAP
        bottom = BOARD_Y + GRID_PIXEL + (GRID_SIZE - 1) * TILE_GAP
        if mx < BOARD_X or mx >= right or my < BOARD_Y or my >= bottom:
            return None
        col = (mx - BOARD_X) // (PIECE_SIZE + TILE_GAP)
        row = (my - BOARD_Y) // (PIECE_SIZE + TILE_GAP)
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return self.puzzle.grid_to_position(row, col)
        return None

    def _get_progress(self) -> float:
        correct = sum(1 for i, p in enumerate(self.puzzle.arrangement) if p == i)
        return correct / NUM_PIECES

    # ── Draw: background + panel shell ───────────────────────────────────────
    def _draw_background(self):
        self.screen.fill(C_BG)
        panel_rect = pygame.Rect(
            BOARD_X + GRID_PIXEL + BOARD_PAD, BOARD_PAD,
            PANEL_WIDTH, WINDOW_H - BOARD_PAD * 2
        )
        draw_rounded_rect(self.screen, C_PANEL, panel_rect, 14, 1, C_PANEL_BORDER)

    # ── Draw: board ───────────────────────────────────────────────────────────
    def _draw_board(self):
        board_w = GRID_PIXEL + (GRID_SIZE - 1) * TILE_GAP
        board_h = board_w
        shadow  = pygame.Rect(BOARD_X - 2, BOARD_Y + 5, board_w + 4, board_h + 2)
        pygame.draw.rect(self.screen, (6, 8, 14), shadow, border_radius=14)
        bg = pygame.Rect(BOARD_X - 3, BOARD_Y - 3, board_w + 6, board_h + 6)
        draw_rounded_rect(self.screen, (22, 26, 42), bg, 14, 1, C_BORDER)

    # ── Draw: tiles ───────────────────────────────────────────────────────────
    def _draw_tiles(self):
        for pos in range(NUM_PIECES):
            if pos == self.dragging_pos:
                continue
            rect        = self._tile_rect(pos)
            piece_index = self.puzzle.arrangement[pos]
            piece_surf  = self.piece_surfaces[piece_index]

            if self.is_solved:
                t = (math.sin(self._pulse + pos * 0.6) + 1) / 2
                border_color = lerp_color(C_BORDER_SOLVED, C_GREEN_DIM, t)
            elif piece_index == pos:
                border_color = C_BORDER_SOLVED
            else:
                border_color = C_BORDER

            if piece_surf:
                self.screen.blit(piece_surf, rect)
            pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=4)

    # ── Draw: drag ghost ──────────────────────────────────────────────────────
    def _draw_drag(self):
        if self.dragging_pos is None or self.drag_surface is None:
            return
        mx, my = self.mouse_pos
        dx     = mx - self.drag_offset[0]
        dy     = my - self.drag_offset[1]

        shadow = pygame.Surface((PIECE_SIZE, PIECE_SIZE), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 70))
        self.screen.blit(shadow, (dx + 8, dy + 8))

        scaled = pygame.transform.scale(
            self.drag_surface, (int(PIECE_SIZE * 1.07), int(PIECE_SIZE * 1.07)))
        ox = (scaled.get_width()  - PIECE_SIZE) // 2
        oy = (scaled.get_height() - PIECE_SIZE) // 2
        self.screen.blit(scaled, (dx - ox, dy - oy))
        r = pygame.Rect(dx - ox, dy - oy, scaled.get_width(), scaled.get_height())
        pygame.draw.rect(self.screen, C_BORDER_DRAG, r, 2, border_radius=6)

    # ── Draw: preview overlay ─────────────────────────────────────────────────
    def _draw_preview(self):
        if self.show_preview and self.preview_surface:
            self.screen.blit(self.preview_surface, (BOARD_X, BOARD_Y))
            dark = pygame.Surface((GRID_PIXEL, GRID_PIXEL), pygame.SRCALPHA)
            dark.fill((0, 0, 0, 35))
            self.screen.blit(dark, (BOARD_X, BOARD_Y))
            lbl = self.f_medium.render("PREVIEW", True, (255, 255, 255))
            self.screen.blit(lbl, (BOARD_X + (GRID_PIXEL - lbl.get_width()) // 2, BOARD_Y + 12))

    # ── Draw: stat card ───────────────────────────────────────────────────────
    def _draw_card(self, rect: pygame.Rect, label: str, value: str):
        draw_rounded_rect(self.screen, C_CARD, rect, BR, 1, C_CARD_BORDER)
        lbl = self.f_label.render(label, True, C_TEXT_LABEL)
        val = self.f_value.render(value, True, C_TEXT_PRIMARY)
        self.screen.blit(lbl, (rect.x + 14, rect.y + 10))
        self.screen.blit(val, (rect.x + 14, rect.y + 30))

    # ── Draw: progress bar ────────────────────────────────────────────────────
    def _draw_progress(self, dt):
        target = self._get_progress()
        self._progress_t += (target - self._progress_t) * min(dt * 6, 1.0)

        correct = int(round(self._progress_t * NUM_PIECES))
        lbl = self.f_label.render(f"Progress  {correct}/9", True, C_TEXT_LABEL)
        self.screen.blit(lbl, self._prog_label_pos)

        draw_rounded_rect(self.screen, C_PROGRESS_BG, self._prog_rect, 5)
        fill_w = int(self._prog_rect.width * self._progress_t)
        if fill_w > 0:
            fill = pygame.Rect(
                self._prog_rect.x, self._prog_rect.y,
                fill_w, self._prog_rect.height
            )
            draw_rounded_rect(self.screen, C_PROGRESS_FILL, fill, 5)

    # ── Draw: full panel ──────────────────────────────────────────────────────
    def _draw_panel(self, dt):
        # Title
        title = self.f_title.render("Puzzle", True, C_TEXT_PRIMARY)
        self.screen.blit(title, (self._title_pos[0] - title.get_width() // 2,
                                  self._title_pos[1]))

        # Dividers
        pygame.draw.line(self.screen, C_CARD_BORDER, *self._divider1, 1)

        # Time & moves cards
        elapsed = (self.completion_time or time.time()) - self.start_time
        m, s    = divmod(int(elapsed), 60)
        self._draw_card(self._card_time,  "Time",  f"{m:02d}:{s:02d}")
        self._draw_card(self._card_moves, "Moves", str(self.move_count))

        # Progress bar
        self._draw_progress(dt)

        # Buttons
        self.btn_preview.draw(self.screen)
        self.btn_restart.draw(self.screen)

        # Divider 2
        pygame.draw.line(self.screen, C_CARD_BORDER, *self._divider2, 1)

        # Instructions
        lines = [
            "Drag & drop tiles to swap.",
            "Hold Preview to see original.",
            "R = restart   |   Q = quit",
        ]
        iy = self._instr_y
        cx = self._instr_cx
        for line in lines:
            s = self.f_small.render(line, True, C_TEXT_SECONDARY)
            self.screen.blit(s, (cx - s.get_width() // 2, iy))
            iy += 20

    # ── Draw: victory overlay ─────────────────────────────────────────────────
    def _draw_victory(self):
        self._victory_alpha = min(190, self._victory_alpha + 5)
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((*C_OVERLAY, self._victory_alpha))
        self.screen.blit(overlay, (0, 0))

        for p in self._confetti:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["rot"] += p["rs"]
            if p["y"] > WINDOW_H + 20:
                p["y"] = float(np.random.randint(-60, -10))
                p["x"] = float(np.random.randint(0, WINDOW_W))
            s  = p["size"]
            cx = int(p["x"]); cy = int(p["y"])
            pts = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
            pygame.draw.polygon(self.screen, p["color"], pts)

        cxv = WINDOW_W // 2
        ty  = WINDOW_H // 2 - 90
        t   = self.f_huge.render("Puzzle Solved!", True, C_GREEN)
        self.screen.blit(t, (cxv - t.get_width() // 2, ty))

        elapsed = (self.completion_time or 0) - self.start_time
        m, s    = divmod(int(elapsed), 60)
        for line in [f"Time: {m:02d}:{s:02d}", f"Moves: {self.move_count}"]:
            ty += 65
            surf = self.f_medium.render(line, True, C_TEXT_PRIMARY)
            self.screen.blit(surf, (cxv - surf.get_width() // 2, ty))

        ty += 55
        hint = self.f_small.render(
            "Press  R  to play again   |   Q  to quit", True, C_TEXT_SECONDARY)
        self.screen.blit(hint, (cxv - hint.get_width() // 2, ty))

    # ── Draw: launch fade-in ──────────────────────────────────────────────────
    def _draw_fadein(self):
        if self._fade_alpha > 0:
            fade = pygame.Surface((WINDOW_W, WINDOW_H))
            fade.fill(C_BG)
            fade.set_alpha(self._fade_alpha)
            self.screen.blit(fade, (0, 0))
            self._fade_alpha = max(0, self._fade_alpha - 14)

    # ── Events ────────────────────────────────────────────────────────────────
    def handle_events(self, dt) -> bool:
        mp = pygame.mouse.get_pos()
        self.mouse_pos = mp
        self.btn_preview.update(mp, dt)
        self.btn_restart.update(mp, dt)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                if event.key == pygame.K_r:
                    self._restart()
                if event.key == pygame.K_p:
                    self.show_preview = not self.show_preview

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_preview.is_clicked(event):
                    self.show_preview = True
                elif self.btn_restart.is_clicked(event):
                    self._restart()
                elif not self.is_solved:
                    self._mouse_down(event)

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.show_preview = False
                self._mouse_up(event)

        return True

    def _mouse_down(self, event):
        pos = self._pos_from_mouse(*event.pos)
        if pos is not None:
            tr = self._tile_rect(pos)
            pi = self.puzzle.arrangement[pos]
            self.drag_surface = self.piece_surfaces[pi]
            self.dragging_pos = pos
            self.drag_offset  = (event.pos[0] - tr.x, event.pos[1] - tr.y)

    def _mouse_up(self, event):
        if self.dragging_pos is None:
            return
        target = self._pos_from_mouse(*event.pos)
        if target is not None and target != self.dragging_pos:
            self.puzzle.swap_pieces(self.dragging_pos, target)
            self.move_count += 1
            if self.puzzle.is_solved() and not self.is_solved:
                self.is_solved       = True
                self.completion_time = time.time()
                self._victory_alpha  = 0
                self._init_confetti()
        self.dragging_pos = None
        self.drag_surface = None

    def _restart(self):
        self.puzzle.shuffle()
        self.piece_surfaces  = self._precompute_piece_surfaces()
        self.move_count      = 0
        self.start_time      = time.time()
        self.is_solved       = False
        self.completion_time = None
        self._victory_alpha  = 0
        self.dragging_pos    = None
        self.drag_surface    = None
        self.show_preview    = False
        self._progress_t     = self._get_progress()
        self._init_confetti()

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            self._pulse += 0.04

            running = self.handle_events(dt)

            self._draw_background()
            self._draw_board()
            self._draw_tiles()
            self._draw_panel(dt)
            self._draw_drag()

            if self.show_preview:
                self._draw_preview()
            if self.is_solved:
                self._draw_victory()

            self._draw_fadein()   # fades out after ~18 frames — hides any launch stutter
            pygame.display.flip()

        pygame.quit()