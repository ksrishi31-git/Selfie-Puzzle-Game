"""
puzzle.py
---------
Handles all puzzle logic:
- Splitting the captured image into a 3x3 grid of pieces.
- Storing the correct (solved) arrangement.
- Shuffling the pieces to create a solvable puzzle.
- Checking for puzzle completion.
"""

import numpy as np
import random
from typing import List, Optional, Tuple

GRID_SIZE = 3
NUM_PIECES = GRID_SIZE * GRID_SIZE


class PuzzlePiece:
    def __init__(self, image: np.ndarray, correct_index: int):
        self.image = image
        self.correct_index = correct_index


class Puzzle:
    def __init__(self):
        self.pieces: List[PuzzlePiece] = []
        self.arrangement: List[int] = list(range(NUM_PIECES))
        self.piece_size: Tuple[int, int] = (0, 0)
        self.source_image: Optional[np.ndarray] = None

    def create_from_image(self, image: np.ndarray, piece_size: int = 160):
        import cv2

        target_w = piece_size * GRID_SIZE
        target_h = piece_size * GRID_SIZE

        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        self.source_image = resized
        self.piece_size = (piece_size, piece_size)
        self.pieces = []

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                y1 = row * piece_size
                y2 = (row + 1) * piece_size
                x1 = col * piece_size
                x2 = (col + 1) * piece_size

                piece_img = resized[y1:y2, x1:x2].copy()
                correct_index = row * GRID_SIZE + col
                self.pieces.append(PuzzlePiece(piece_img, correct_index))

        self.arrangement = list(range(NUM_PIECES))

    def shuffle(self, max_attempts: int = 1000):
        solved = list(range(NUM_PIECES))
        for _ in range(max_attempts):
            random.shuffle(self.arrangement)
            if self.arrangement != solved:
                return

        if len(self.arrangement) >= 2:
            self.arrangement[0], self.arrangement[1] = self.arrangement[1], self.arrangement[0]

    def swap_pieces(self, pos_a: int, pos_b: int):
        if 0 <= pos_a < NUM_PIECES and 0 <= pos_b < NUM_PIECES:
            self.arrangement[pos_a], self.arrangement[pos_b] = (
                self.arrangement[pos_b],
                self.arrangement[pos_a],
            )

    def is_solved(self) -> bool:
        return self.arrangement == list(range(NUM_PIECES))

    def get_piece_at(self, position: int) -> Optional[PuzzlePiece]:
        if 0 <= position < NUM_PIECES:
            piece_index = self.arrangement[position]
            return self.pieces[piece_index]
        return None

    def get_piece_image_at(self, position: int) -> Optional[np.ndarray]:
        piece = self.get_piece_at(position)
        return piece.image if piece else None

    def get_full_preview_image(self) -> Optional[np.ndarray]:
        return self.source_image

    def position_to_grid(self, position: int) -> Tuple[int, int]:
        return divmod(position, GRID_SIZE)

    def grid_to_position(self, row: int, col: int) -> int:
        return row * GRID_SIZE + col
