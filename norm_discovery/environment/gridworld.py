"""Grid layout and generic geometric primitives for the shopping gridworld.

A layout is a rectangular grid of free cells and shelf cells (obstacles), plus
named special locations (entrance, exit, cart station, cart-return area) and
items stored on shelf cells.  Items are picked up from a free cell that is
4-adjacent to the item's shelf cell.

Geometric primitives are *generic* relations computed for every free cell
(neighbouring shelves, free-space width).  They form the spatial refinement
vocabulary.  Ground-truth aisle membership is stored on the layout only for
evaluation / plotting and is never exposed to any learner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]

# y grows downward (row index), x grows to the right (column index).
DIRS: Dict[str, Cell] = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
DIR_ORDER = ("N", "S", "E", "W")

GEOM_BOOL = ("free", "adjShelf", "shelfN", "shelfS", "shelfE", "shelfW")
GEOM_NUM = ("freeWidthH", "freeWidthV")


@dataclass
class Layout:
    name: str
    width: int
    height: int
    shelves: frozenset
    entrance: Cell
    exit: Cell
    cart_station: Cell
    return_area: Optional[Cell]
    items: Dict[str, Cell]
    aisles: Dict[str, List[Cell]] = field(default_factory=dict)  # ground truth (evaluation only)
    standalone_shelves: List[Cell] = field(default_factory=list)

    def __post_init__(self):
        self.free_cells: List[Cell] = [
            (x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in self.shelves
        ]
        self.cell_index: Dict[Cell, int] = {c: i for i, c in enumerate(self.free_cells)}
        for name, c in self.items.items():
            if c not in self.shelves:
                raise ValueError(f"item {name} must be on a shelf cell, got {c}")
        for c in (self.entrance, self.exit, self.cart_station) + ((self.return_area,) if self.return_area else ()):
            if c in self.shelves:
                raise ValueError(f"special cell {c} is a shelf")
        self.geometry = compute_geometry(self)
        # move table: move_to[d][i] = index of destination free cell or -1 if blocked
        self.move_to = {}
        for d, (dx, dy) in DIRS.items():
            arr = np.full(len(self.free_cells), -1, dtype=np.int64)
            for i, (x, y) in enumerate(self.free_cells):
                arr[i] = self.cell_index.get((x + dx, y + dy), -1)
            self.move_to[d] = arr

    # ------------------------------------------------------------------
    @classmethod
    def from_ascii(cls, name: str, rows: List[str], items: Dict[str, Cell],
                   aisles: Optional[Dict[str, List[Cell]]] = None,
                   standalone: Optional[List[Cell]] = None) -> "Layout":
        """'.' free, '#' shelf, 'E' entrance, 'X' exit, 'C' cart station, 'R' cart return."""
        height, width = len(rows), len(rows[0])
        shelves, special = set(), {}
        for y, row in enumerate(rows):
            if len(row) != width:
                raise ValueError("ragged layout")
            for x, ch in enumerate(row):
                if ch == "#":
                    shelves.add((x, y))
                elif ch in "EXCR":
                    special[ch] = (x, y)
                elif ch != ".":
                    raise ValueError(f"unknown layout char {ch!r}")
        return cls(name=name, width=width, height=height, shelves=frozenset(shelves),
                   entrance=special["E"], exit=special["X"], cart_station=special["C"],
                   return_area=special.get("R"), items=dict(items), aisles=dict(aisles or {}),
                   standalone_shelves=list(standalone or []))

    def in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.width and 0 <= c[1] < self.height

    def is_free(self, c: Cell) -> bool:
        return self.in_bounds(c) and c not in self.shelves

    def is_shelf(self, c: Cell) -> bool:
        return c in self.shelves

    def item_access_cells(self, item: str) -> List[Cell]:
        x, y = self.items[item]
        return [(x + dx, y + dy) for dx, dy in DIRS.values() if self.is_free((x + dx, y + dy))]

    def ascii(self) -> str:
        rows = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                c = (x, y)
                ch = "#" if c in self.shelves else "."
                for k, v in (("E", self.entrance), ("X", self.exit), ("C", self.cart_station), ("R", self.return_area)):
                    if v == c:
                        ch = k
                if c in self.items.values():
                    ch = "i"
                row += ch
            rows.append(row)
        return "\n".join(rows)


def compute_geometry(layout: Layout) -> Dict[str, np.ndarray]:
    """Generic geometric relations for every free cell (indexed like layout.free_cells)."""
    n = len(layout.free_cells)
    g = {k: np.zeros(n, dtype=bool) for k in GEOM_BOOL}
    g.update({k: np.zeros(n, dtype=np.int64) for k in GEOM_NUM})
    for i, (x, y) in enumerate(layout.free_cells):
        g["free"][i] = True
        g["shelfN"][i] = layout.is_shelf((x, y - 1))
        g["shelfS"][i] = layout.is_shelf((x, y + 1))
        g["shelfE"][i] = layout.is_shelf((x + 1, y))
        g["shelfW"][i] = layout.is_shelf((x - 1, y))
        g["adjShelf"][i] = g["shelfN"][i] or g["shelfS"][i] or g["shelfE"][i] or g["shelfW"][i]
        # width of the maximal horizontal / vertical run of free cells through (x,y)
        w = 1
        xx = x - 1
        while layout.is_free((xx, y)):
            w += 1
            xx -= 1
        xx = x + 1
        while layout.is_free((xx, y)):
            w += 1
            xx += 1
        g["freeWidthH"][i] = w
        h = 1
        yy = y - 1
        while layout.is_free((x, yy)):
            h += 1
            yy -= 1
        yy = y + 1
        while layout.is_free((x, yy)):
            h += 1
            yy += 1
        g["freeWidthV"][i] = h
    return g
