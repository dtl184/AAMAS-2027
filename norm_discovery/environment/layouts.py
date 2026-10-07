"""Reconstructed store layouts (Fig. 1 and Fig. 2 of the paper)."""
from __future__ import annotations

from .gridworld import Layout

# Experiment 1 (cart-use norm), after Fig. 1: 10 x 6 grid, one shelf block S,
# cart station C next to the entrance E, cart-return area R, exit X.
EXP1_ROWS = [
    "..........",
    "R.........",
    "..........",
    "..#####...",
    "..........",
    "EC.......X",
]
EXP1_ITEMS = {"apple": (2, 3), "bread": (3, 3), "milk": (5, 3), "eggs": (6, 3)}

# Experiment 2 (aisle-use norm), after Fig. 2: 12 x 8 grid.  Three dead-end aisles
# (columns 3, 6, 9; rows 0-2) between shelf blocks, two standalone displays.
EXP2_ROWS = [
    "..#.##.##.#.",
    "..#.##.##.#.",
    "..#.##.##.#.",
    "............",
    "............",
    "......#..#..",
    "............",
    "EC.........X",
]
EXP2_ITEMS = {
    "a1": (4, 1),   # reachable only from aisle A1 cell (3,1)
    "a2": (5, 1),   # reachable only from aisle A2 cell (6,1)
    "a3": (8, 1),   # reachable only from aisle A3 cell (9,1)   -- never used in training
    "s1": (6, 5),   # standalone display
    "s2": (9, 5),   # second standalone display
}
EXP2_AISLES = {"A1": [(3, 0), (3, 1), (3, 2)], "A2": [(6, 0), (6, 1), (6, 2)], "A3": [(9, 0), (9, 1), (9, 2)]}


def exp1_layout() -> Layout:
    return Layout.from_ascii("store_exp1", EXP1_ROWS, EXP1_ITEMS)


def exp2_layout() -> Layout:
    return Layout.from_ascii("store_exp2", EXP2_ROWS, EXP2_ITEMS, aisles=EXP2_AISLES,
                             standalone=[(6, 5), (9, 5)])
