"""Deterministic shopping planning model M = <S, A, T, C>.

Low-level state  s = (pos, cart, mask, hand)
  pos   index of the agent's free cell (Layout.free_cells)
  cart  CART_STATION (-2): cart sits at the cart station
        CART_HELD    (-1): agent is pushing the cart
        k >= 0           : cart parked at free cell k
  mask  bitmask over the task's required items already collected
  hand  1 if the agent is carrying an item by hand (only with hand_capacity=1; always 0 by default)

Primitive actions (all cost 1):
  N, S, E, W          move to a free 4-neighbour
  PICKUP_CART         at the station (cart at station) or on the cell where the cart is
                      parked; hand-carried item is transferred into the cart (hand := 0)
  LEAVE_CART          park the held cart on the current cell (at the station cell it is
                      put back into the station)
  PICKUP_ITEM(i)      adjacent to item i's shelf, i required and not yet collected, and the
                      cart is in use: held, or parked within Manhattan distance `cart_reach`
                      (default 2) of the agent, i.e. the item can be put into the nearby cart.
                      Optional mechanic (hand_capacity=1, off by default): with no usable cart
                      and an empty hand, the item is carried by hand (hand := 1).
  EXIT                at the exit cell with all required items collected (and, if the
                      domain sets exit_requires_cart, while holding the cart). Terminal.

Items are counted as collected (purchased) once picked; the cart only matters for
carrying capacity.  This is a documented simplification (see README).

The initial temporal abstraction collapses PICKUP_CART, LEAVE_CART and PICKUP_ITEM
into a single INTERACT label (see abstractions/base.py); the dynamics themselves
always use the primitive actions.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .gridworld import DIR_ORDER, Layout

CART_STATION = -2
CART_HELD = -1

ACT_N, ACT_S, ACT_E, ACT_W, ACT_PICKUP_CART, ACT_LEAVE_CART, ACT_EXIT = range(7)
ACT_ITEM0 = 7  # PICKUP_ITEM for global item k has code ACT_ITEM0 + k
MOVE_CODES = {"N": ACT_N, "S": ACT_S, "E": ACT_E, "W": ACT_W}
BASE_ACTION_NAMES = ["N", "S", "E", "W", "PICKUP_CART", "LEAVE_CART", "EXIT"]

State = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Task:
    """A task context (s0, g): start state and the set of required items."""
    items: Tuple[str, ...]
    start: State

    @property
    def key(self) -> str:
        return f"items={'+'.join(self.items) or '-'}|start={self.start}"


@dataclass
class Graph:
    """All states reachable from a task's start state and all primitive transitions."""
    task: Task
    states: List[State]
    index: Dict[State, int]
    S_pos: np.ndarray
    S_cart: np.ndarray
    S_mask: np.ndarray
    S_hand: np.ndarray
    E_src: np.ndarray
    E_dst: np.ndarray  # -1 for the terminal EXIT transition
    E_act: np.ndarray
    _batch: object = field(default=None, repr=False)

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_edges(self) -> int:
        return len(self.E_src)


class ShoppingDomain:
    def __init__(self, name: str, layout: Layout, exit_requires_cart: bool,
                 vocab_levels: Dict[int, Sequence[str]], mlci_families: Sequence[str],
                 meta: Optional[dict] = None, cart_reach: int = 2, hand_capacity: int = 0):
        self.name = name
        self.layout = layout
        self.exit_requires_cart = exit_requires_cart
        self.cart_reach = int(cart_reach)
        self.hand_capacity = int(hand_capacity)
        self.vocab_levels = {int(k): tuple(v) for k, v in vocab_levels.items()}
        self.mlci_families = tuple(mlci_families)
        self.meta = dict(meta or {})
        self.item_names: List[str] = sorted(layout.items)
        self.item_code = {n: ACT_ITEM0 + k for k, n in enumerate(self.item_names)}
        L = layout
        self.station_idx = L.cell_index[L.cart_station]
        self.exit_idx = L.cell_index[L.exit]
        self.entrance_idx = L.cell_index[L.entrance]
        # cells from which each item can be picked
        self.item_access = {n: {L.cell_index[c] for c in L.item_access_cells(n)} for n in self.item_names}
        self._graphs: Dict[str, Graph] = {}

    # ------------------------------------------------------------------ naming
    def action_name(self, a: int) -> str:
        if a < ACT_ITEM0:
            return BASE_ACTION_NAMES[a]
        return f"PICKUP_ITEM({self.item_names[a - ACT_ITEM0]})"

    def action_code(self, name: str) -> int:
        if name in BASE_ACTION_NAMES:
            return BASE_ACTION_NAMES.index(name)
        if name.startswith("PICKUP_ITEM(") and name.endswith(")"):
            return self.item_code[name[len("PICKUP_ITEM("):-1]]
        raise KeyError(name)

    def cell(self, pos: int):
        return self.layout.free_cells[pos]

    def describe_state(self, s: State, task: Task) -> str:
        pos, cart, mask, hand = s
        if cart == CART_HELD:
            c = "held"
        elif cart == CART_STATION:
            c = "station"
        else:
            c = f"parked@{self.cell(cart)}"
        got = [it for k, it in enumerate(task.items) if mask >> k & 1]
        return f"pos={self.cell(pos)} cart={c} items={got} hand={hand}"

    # ------------------------------------------------------------------ tasks
    def make_task(self, items: Sequence[str], start: Optional[State] = None) -> Task:
        items = tuple(sorted(items))
        for it in items:
            if it not in self.layout.items:
                raise KeyError(it)
        if start is None:
            start = (self.entrance_idx, CART_STATION, 0, 0)
        return Task(items=items, start=start)

    # ------------------------------------------------------------------ dynamics
    def successors(self, s: State, task: Task) -> List[Tuple[int, Optional[State]]]:
        """Available (action, next_state) pairs in canonical order; next_state None = terminal."""
        pos, cart, mask, hand = s
        out: List[Tuple[int, Optional[State]]] = []
        L = self.layout
        for d in DIR_ORDER:
            nxt = int(L.move_to[d][pos])
            if nxt >= 0:
                out.append((MOVE_CODES[d], (nxt, cart, mask, hand)))
        if cart != CART_HELD and ((cart == CART_STATION and pos == self.station_idx) or cart == pos):
            out.append((ACT_PICKUP_CART, (pos, CART_HELD, mask, 0)))
        if cart == CART_HELD:
            parked = CART_STATION if pos == self.station_idx else pos
            out.append((ACT_LEAVE_CART, (pos, parked, mask, hand)))
        full = (1 << len(task.items)) - 1
        if pos == self.exit_idx and mask == full and (not self.exit_requires_cart or cart == CART_HELD):
            out.append((ACT_EXIT, None))
        if cart == CART_HELD:
            usable = True
        elif cart >= 0:
            (ax, ay), (cx, cy) = L.free_cells[pos], L.free_cells[cart]
            usable = abs(ax - cx) + abs(ay - cy) <= self.cart_reach
        else:
            usable = False
        for k, it in enumerate(task.items):
            if mask >> k & 1 or pos not in self.item_access[it]:
                continue
            if usable:
                out.append((self.item_code[it], (pos, cart, mask | (1 << k), hand)))
            elif self.hand_capacity > 0 and hand == 0:
                out.append((self.item_code[it], (pos, cart, mask | (1 << k), 1)))
        return out

    def step(self, s: State, a: int, task: Task) -> Optional[State]:
        for act, nxt in self.successors(s, task):
            if act == a:
                return nxt
        raise ValueError(f"action {self.action_name(a)} not available in {self.describe_state(s, task)}")

    def is_available(self, s: State, a: int, task: Task) -> bool:
        return any(act == a for act, _ in self.successors(s, task))

    # ------------------------------------------------------------------ graph
    def graph(self, task: Task) -> Graph:
        g = self._graphs.get(task.key)
        if g is None:
            g = self._build_graph(task)
            self._graphs[task.key] = g
        return g

    def _build_graph(self, task: Task) -> Graph:
        index: Dict[State, int] = {task.start: 0}
        states: List[State] = [task.start]
        src, dst, act = [], [], []
        q = deque([task.start])
        while q:
            s = q.popleft()
            i = index[s]
            for a, nxt in self.successors(s, task):
                src.append(i)
                act.append(a)
                if nxt is None:
                    dst.append(-1)
                    continue
                j = index.get(nxt)
                if j is None:
                    j = len(states)
                    index[nxt] = j
                    states.append(nxt)
                    q.append(nxt)
                dst.append(j)
        S = np.array(states, dtype=np.int64)
        return Graph(task=task, states=states, index=index, S_pos=S[:, 0], S_cart=S[:, 1], S_mask=S[:, 2],
                     S_hand=S[:, 3], E_src=np.array(src, dtype=np.int64), E_dst=np.array(dst, dtype=np.int64),
                     E_act=np.array(act, dtype=np.int64))
