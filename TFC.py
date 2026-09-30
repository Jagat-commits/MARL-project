"""
city_traffic.py - interactive 2D city traffic sandbox (environment + visualiser)

Run:  python city_traffic.py

Where to plug things in:
    City.policy        signal controller -> replace with your MARL policy
                       signature: policy(city, light) -> 0 (N-S green) or 1 (E-W green)
    City.observe(l)    state for the RL agent: stopped-queue length on NS / EW
    City.bfs(...)      router -> replace with A* / congestion-aware routing

Mouse: left-click/drag paints with the current tool (keys 1-4), right-click a car
selects it, right-click a cell re-routes the selected car there.
Keys: Space pause, D auto-spawn, T spawn car, N/X add/remove car at cursor, M signal mode,
      R new layout, C clear, S/L save/load, G grid, P paths, +/- speed, [ ] spawn rate, Esc quit
"""
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import pygame

# ---------------------------------------------------------------- config
COLS, ROWS = 22, 15
CELL, MARGIN, SIDEBAR = 36, 16, 280
SAVE_FILE = Path("city_layout.json")

BUILDING, ROAD, JUNCTION, DEST, BLOCK, SPAWN = range(6)          # cell types
CELL_NAMES = ["Building", "Road", "Junction", "Destination", "Roadblock", "Spawner"]
TOOLS = [("Roadblock", BLOCK), ("Clear", ROAD), ("Destination", DEST), ("Spawner", SPAWN)]

NONE, EAST, WEST, SOUTH, NORTH, ANY = range(6)                   # lane codes (ANY = junction)
DIRS = {EAST: (1, 0), WEST: (-1, 0), SOUTH: (0, 1), NORTH: (0, -1)}
CODE = {v: k for k, v in DIRS.items()}
OPPOSITE = {EAST: WEST, WEST: EAST, SOUTH: NORTH, NORTH: SOUTH}
LANE_NAMES = ["-", "eastbound", "westbound", "southbound", "northbound", "junction"]
LEFT_HAND = True                # India drives on the left

H_ROADS = (2, 9)                # top row of each 2-wide horizontal road
V_ROADS = (2, 10, 18)           # left column of each 2-wide vertical road
GREEN_TICKS, MAX_GREEN, MIN_GREEN, YELLOW_TICKS = 10, 20, 4, 2
PATIENCE = 3                    # ticks stuck behind a car before trying to go around
MAX_VEHICLES = 45
QUEUE_LEN = 4                   # cells before a junction that count as the queue

BG, PANEL = (15, 17, 24), (24, 27, 38)
ROAD_A, ROAD_B, JUNC_C = (44, 48, 62), (47, 52, 67), (56, 61, 80)
GRID_LINE, LANE_LINE, ARROW_C = (52, 58, 78), (196, 170, 70), (70, 77, 100)
BLD_TONES = [((92, 108, 153), (128, 145, 195)), ((84, 120, 140), (120, 165, 185)),
             ((110, 96, 150), (150, 135, 195))]
BLD_SH, WINDOW_C = (12, 14, 20), (176, 156, 96)
GREEN, YELLOW, RED, ORANGE = (52, 211, 153), (251, 191, 36), (239, 68, 68), (249, 115, 22)
TEXT, MUTED, ACCENT = (226, 232, 240), (140, 150, 170), (129, 140, 248)
TOOL_COLORS = [ORANGE, MUTED, GREEN, YELLOW]
CAR_COLORS = [(239, 68, 68), (59, 130, 246), (168, 85, 247), (236, 72, 153),
              (20, 184, 166), (249, 115, 22), (132, 204, 22), (14, 165, 233)]


# ---------------------------------------------------------------- environment
class Vehicle:
    def __init__(self, vid, x, y, dest, path):
        self.id, self.x, self.y = vid, x, y
        self.color = CAR_COLORS[vid % len(CAR_COLORS)]
        self.dest, self.path = dest, path
        self.wait = 0                       # ticks in a row not moving
        self.total_wait = 0                 # ticks not moving over the whole trip (metric to minimise)
        self.custom_dest = False            # set by right-click re-route
        self.hx, self.hy = 1, 0             # heading
        self.rx, self.ry = float(x), float(y)   # smoothed position for drawing


class Light:
    """One signalised junction (2x2 cells). phase 0 = N-S green, phase 1 = E-W green."""

    def __init__(self, cells, phase, timer):
        self.cells = cells
        self.approaches = []                # (x, y, dir): last road cell before the junction
        self.phase = self.next_phase = phase
        self.timer = timer                  # ticks since the phase started
        self.yellow = 0                     # > 0 while switching, nobody may enter

    def allows(self, code):
        return self.yellow == 0 and ((code in (SOUTH, NORTH)) == (self.phase == 0))

    def request(self, phase):
        """The agent's action. Ignored while switching or before MIN_GREEN."""
        if phase != self.phase and self.yellow == 0 and self.timer >= MIN_GREEN:
            self.next_phase, self.yellow = phase, YELLOW_TICKS

    def tick(self):
        self.timer += 1
        if self.yellow:
            self.yellow -= 1
            if self.yellow == 0:
                self.phase, self.timer = self.next_phase, 0


# placeholder signal policies (replace with your trained MARL policy)
def fixed_time_policy(city, light):
    return 1 - light.phase if light.timer >= GREEN_TICKS else light.phase


def adaptive_policy(city, light):
    q = city.observe(light)
    cur, other = (q["NS"], q["EW"]) if light.phase == 0 else (q["EW"], q["NS"])
    if light.timer >= MAX_GREEN or (light.timer >= MIN_GREEN and other > cur):
        return 1 - light.phase
    return light.phase


POLICIES = [("Fixed-time", fixed_time_policy), ("Adaptive (queue)", adaptive_policy)]


class City:
    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=np.int8)   # cell type
        self.lane = np.zeros((ROWS, COLS), dtype=np.int8)   # lane direction (fixed road network)
        self.vehicles, self.lights = [], []
        self.cell_light = {}                # junction cell -> its Light
        self.flow = {}                      # junction cell -> allowed directions
        self.policy = fixed_time_policy
        self.spawn_rate = 0.35
        self.tick = self.arrived = self.arrived_wait = self.next_id = 0

    # ---- queries ----
    def in_bounds(self, x, y):
        return 0 <= x < COLS and 0 <= y < ROWS

    def is_walkable(self, x, y):
        return self.in_bounds(x, y) and self.grid[y, x] not in (BUILDING, BLOCK)

    def can_move(self, a, b):
        """Lane rules: stay in your lane, junction cells allow turns."""
        if not self.is_walkable(*b):
            return False
        code = CODE[(b[0] - a[0], b[1] - a[1])]
        la, lb = int(self.lane[a[1], a[0]]), int(self.lane[b[1], b[0]])
        if la == ANY and lb == ANY:         # inside a junction: one-way ring, no head-ons
            return code in self.flow[a]
        return la in (code, ANY) and lb in (code, ANY)

    def neighbors(self, x, y, strict=True):
        for dx, dy in DIRS.values():
            nb = (x + dx, y + dy)
            if self.can_move((x, y), nb) if strict else self.is_walkable(*nb):
                yield nb

    def vehicle_at(self, x, y):
        return next((v for v in self.vehicles if v.x == x and v.y == y), None)

    def get_vehicle(self, vid):
        return next((v for v in self.vehicles if v.id == vid), None)

    def observe(self, light):
        """What the agent sees: number of stopped cars queued on each axis."""
        q = {"NS": 0, "EW": 0}
        for ax, ay, code in light.approaches:
            dx, dy = DIRS[code]
            for k in range(QUEUE_LEN):
                v = self.vehicle_at(ax - dx * k, ay - dy * k)
                if v and v.wait > 0:
                    q["NS" if code in (SOUTH, NORTH) else "EW"] += 1
        return q

    # ---- layout ----
    def generate_layout(self, seed=None):
        rng = random.Random(seed)
        self.grid[:], self.lane[:] = BUILDING, NONE
        self.vehicles.clear()
        self.lights.clear()
        self.cell_light.clear()
        self.tick = self.arrived = self.arrived_wait = self.next_id = 0

        top, bot = (EAST, WEST) if LEFT_HAND else (WEST, EAST)
        west, east = (NORTH, SOUTH) if LEFT_HAND else (SOUTH, NORTH)
        for y0 in H_ROADS:                  # 2-wide roads, one lane each way
            self.lane[y0, :], self.lane[y0 + 1, :] = top, bot
        for x0 in V_ROADS:
            for y in range(ROWS):
                for off, d in ((0, west), (1, east)):
                    self.lane[y, x0 + off] = ANY if self.lane[y, x0 + off] else d
        self.grid[self.lane > 0] = ROAD
        self.grid[self.lane == ANY] = JUNCTION
        self.flow = {(int(x), int(y)): (top if y in H_ROADS else bot, west if x in V_ROADS else east)
                     for y, x in zip(*np.where(self.lane == ANY))}

        for y0 in H_ROADS:                  # one signal per crossing
            for x0 in V_ROADS:
                cells = [(x0 + i, y0 + j) for j in (0, 1) for i in (0, 1)]
                light = Light(cells, rng.randrange(2), rng.randrange(GREEN_TICKS))
                for cx, cy in cells:
                    for code, (dx, dy) in DIRS.items():
                        px, py = cx - dx, cy - dy
                        if self.in_bounds(px, py) and self.lane[py, px] == code:
                            light.approaches.append((px, py, code))
                    self.cell_light[(cx, cy)] = light
                self.lights.append(light)

        entries, inside = [], []            # spawners on the map edge, destinations inside
        for y in range(ROWS):
            for x in range(COLS):
                code = int(self.lane[y, x])
                if code in DIRS:
                    dx, dy = DIRS[code]
                    (inside if self.in_bounds(x - dx, y - dy) else entries).append((x, y))
        for x, y in entries:
            self.grid[y, x] = SPAWN
        for x, y in rng.sample(inside, min(10, len(inside))):
            self.grid[y, x] = DEST

    def clear(self):
        self.grid[self.grid == BLOCK] = ROAD
        self.grid[self.lane == ANY] = JUNCTION
        self.vehicles.clear()

    def validate_trips(self):
        """After painting, make sure every car still has a valid destination."""
        for v in list(self.vehicles):
            ok = self.is_walkable(*v.dest) if v.custom_dest else self.grid[v.dest[1], v.dest[0]] == DEST
            if not ok:
                pick = self.pick_dest((v.x, v.y))
                if pick:
                    v.dest, v.custom_dest, v.path = pick[0], False, pick[1]
                else:
                    self.vehicles.remove(v)

    # ---- vehicles ----
    def cells_of(self, kind):
        cells = [(int(x), int(y)) for y, x in zip(*np.where(self.grid == kind))]
        random.shuffle(cells)
        return cells

    def pick_dest(self, pos):
        for d in self.cells_of(DEST):
            if d != pos:
                path = self.bfs(pos, d)
                if path is not None:
                    return d, path
        return None

    def add_vehicle(self, x, y):
        if not self.is_walkable(x, y) or self.vehicle_at(x, y):
            return None
        pick = self.pick_dest((x, y))
        if pick is None:
            return None
        v = Vehicle(self.next_id, x, y, *pick)
        v.hx, v.hy = DIRS.get(int(self.lane[y, x]), (1, 0))
        self.next_id += 1
        self.vehicles.append(v)
        return v

    def spawn_vehicle(self):
        for x, y in self.cells_of(SPAWN):
            v = self.add_vehicle(x, y)
            if v:
                return v
        return None

    # ---- routing (placeholder: replace with A* / congestion cost) ----
    def bfs(self, start, goal, blocked=frozenset(), strict=True):
        if start == goal:
            return []
        prev, queue = {start: None}, deque([start])
        while queue:
            cur = queue.popleft()
            if cur == goal:
                break
            for nb in self.neighbors(*cur, strict=strict):
                if nb not in prev and (nb not in blocked or nb == goal):
                    prev[nb] = cur
                    queue.append(nb)
        if goal not in prev:
            return None
        path, cur = [], goal
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    def route(self, start, goal):
        path = self.bfs(start, goal)
        if path is None:                    # no legal route (roadblock) -> illegal U-turn escape
            path = self.bfs(start, goal, strict=False)
        return path

    # ---- simulation ----
    def wait(self, v):
        v.wait += 1
        v.total_wait += 1

    def move(self, v, occ, done, gone):
        """Move one car a single cell. Cars ahead move first so queues flow."""
        if v.id in done:
            return
        done.add(v.id)
        pos = (v.x, v.y)
        if pos == v.dest:
            gone.append(v)
            occ.pop(pos, None)
            return
        if not v.path:
            v.path = self.route(pos, v.dest) or []
            if not v.path:
                return self.wait(v)
        nxt = v.path[0]
        if not self.is_walkable(*nxt):      # roadblock appeared -> replan
            v.path = self.route(pos, v.dest) or []
            return self.wait(v)

        code = CODE[(nxt[0] - pos[0], nxt[1] - pos[1])]
        light = self.cell_light.get(nxt)
        if light and self.lane[pos[1], pos[0]] != ANY:      # about to enter a junction
            if not light.allows(code) or sum(c in occ for c in light.cells) >= 3:
                return self.wait(v)         # red / yellow, or keep one cell free so it can rotate
            exit_cell = next((c for c in v.path[1:] if self.cell_light.get(c) is not light), None)
            if exit_cell in occ:            # don't block the box
                self.move(occ[exit_cell], occ, done, gone)
                if exit_cell in occ:
                    return self.wait(v)

        if nxt in occ:
            self.move(occ[nxt], occ, done, gone)            # let the car ahead go first
        if nxt in occ:
            self.wait(v)
            if v.wait >= PATIENCE:                          # try to go around
                alt = self.bfs(pos, v.dest, set(occ) - {pos})
                if alt:
                    v.path = alt
            return

        del occ[pos]
        occ[nxt] = v
        v.hx, v.hy = nxt[0] - pos[0], nxt[1] - pos[1]
        v.x, v.y = nxt
        v.path.pop(0)
        v.wait = 0

    def step(self, auto=True):
        self.tick += 1
        if auto and len(self.vehicles) < MAX_VEHICLES and random.random() < self.spawn_rate:
            self.spawn_vehicle()
        for light in self.lights:
            light.tick()
            light.request(self.policy(self, light))
        occ = {(v.x, v.y): v for v in self.vehicles}
        done, gone = set(), []
        for v in list(self.vehicles):
            self.move(v, occ, done, gone)
        for v in gone:
            self.arrived += 1
            self.arrived_wait += v.total_wait
            self.vehicles.remove(v)

    # ---- save / load ----
    def save(self):
        data = {"grid": self.grid.tolist(), "vehicles": [[v.x, v.y] for v in self.vehicles]}
        SAVE_FILE.write_text(json.dumps(data))

    def load(self):
        data = json.loads(SAVE_FILE.read_text())
        g = np.array(data["grid"], dtype=np.int8)
        if g.shape != self.grid.shape:
            raise ValueError("grid size mismatch")
        if ((g != BUILDING) != (self.lane > 0)).any():
            raise ValueError("layout does not match road network")
        self.grid = g
        self.vehicles.clear()
        self.tick = self.arrived = self.arrived_wait = 0
        for x, y in data["vehicles"]:
            self.add_vehicle(x, y)


# ---------------------------------------------------------------- visualiser
class App:
    def __init__(self):
        pygame.init()
        self.gw, self.gh = COLS * CELL, ROWS * CELL
        self.W = 2 * MARGIN + self.gw + SIDEBAR
        self.H = max(2 * MARGIN + self.gh + 90, 720)
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("City Traffic Sandbox")
        self.clock = pygame.time.Clock()
        fonts = "segoeui,helvetica,arial,dejavusans"
        self.f_lg = pygame.font.SysFont(fonts, 24, bold=True)
        self.f_md = pygame.font.SysFont(fonts, 16)
        self.f_sm = pygame.font.SysFont(fonts, 13, bold=True)

        self.city = City()
        self.new_layout(7)
        self.tool = 0
        self.policy_idx = 0
        self.selected = None                # id of the selected car
        self.painting = False
        self.paused = False
        self.auto = True
        self.grid_lines = False
        self.show_paths = True
        self.step_ms, self.acc = 220, 0
        self.running = True
        self.toast_text, self.toast_until = "", 0

    def new_layout(self, seed):
        self.city.generate_layout(seed)
        self.selected = None
        for _ in range(6):
            self.city.spawn_vehicle()

    # ---- helpers ----
    def toast(self, msg):
        self.toast_text, self.toast_until = msg, pygame.time.get_ticks() + 2200

    def cell_rect(self, x, y):
        return pygame.Rect(MARGIN + x * CELL, MARGIN + y * CELL, CELL, CELL)

    def mouse_cell(self, pos):
        x, y = (pos[0] - MARGIN) // CELL, (pos[1] - MARGIN) // CELL
        return (x, y) if self.city.in_bounds(x, y) else None

    def text(self, s, pos, font=None, color=TEXT, anchor="topleft"):
        surf = (font or self.f_md).render(s, True, color)
        self.screen.blit(surf, surf.get_rect(**{anchor: pos}))

    # ---- input ----
    def handle_event(self, e):
        if e.type == pygame.QUIT:
            self.running = False
        elif e.type == pygame.KEYDOWN:
            self.on_key(e.key)
        elif e.type == pygame.MOUSEBUTTONDOWN:
            cell = self.mouse_cell(e.pos)
            if cell and e.button == 1:
                self.painting = True
                self.paint(cell)
            elif cell and e.button == 3:
                self.right_click(cell)
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.painting = False
        elif e.type == pygame.MOUSEMOTION and self.painting:
            cell = self.mouse_cell(e.pos)
            if cell:
                self.paint(cell)

    def on_key(self, k):
        c = self.city
        cell = self.mouse_cell(pygame.mouse.get_pos())
        if k == pygame.K_ESCAPE:
            self.running = False
        elif pygame.K_1 <= k <= pygame.K_4:
            self.tool = k - pygame.K_1
        elif k == pygame.K_SPACE:
            self.paused = not self.paused
        elif k == pygame.K_d:
            self.auto = not self.auto
        elif k == pygame.K_g:
            self.grid_lines = not self.grid_lines
        elif k == pygame.K_p:
            self.show_paths = not self.show_paths
        elif k == pygame.K_t and not c.spawn_vehicle():
            self.toast("No free spawner / reachable destination")
        elif k == pygame.K_n and cell and not c.add_vehicle(*cell):
            self.toast("Can't place a car there")
        elif k == pygame.K_x and cell:
            v = c.vehicle_at(*cell)
            if v:
                c.vehicles.remove(v)
                if self.selected == v.id:
                    self.selected = None
        elif k == pygame.K_m:
            self.policy_idx = (self.policy_idx + 1) % len(POLICIES)
            c.policy = POLICIES[self.policy_idx][1]
            self.toast(f"Signals: {POLICIES[self.policy_idx][0]}")
        elif k == pygame.K_r:
            self.new_layout(random.randrange(10_000))
            self.toast("New layout")
        elif k == pygame.K_c:
            c.clear()
            self.selected = None
            self.toast("Roadblocks and cars cleared")
        elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.step_ms = max(50, self.step_ms - 30)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.step_ms = min(600, self.step_ms + 30)
        elif k == pygame.K_LEFTBRACKET:
            c.spawn_rate = max(0.0, round(c.spawn_rate - 0.05, 2))
        elif k == pygame.K_RIGHTBRACKET:
            c.spawn_rate = min(1.0, round(c.spawn_rate + 0.05, 2))
        elif k == pygame.K_s:
            c.save()
            self.toast(f"Saved to {SAVE_FILE}")
        elif k == pygame.K_l:
            try:
                c.load()
                self.selected = None
                self.toast("Layout loaded")
            except Exception as ex:
                self.toast(f"Load failed: {ex}")

    def paint(self, cell):
        x, y = cell
        c = self.city
        value = TOOLS[self.tool][1]
        lane = int(c.lane[y, x])
        if lane == NONE:
            self.toast("Buildings are fixed - paint on roads")
        elif value == BLOCK and c.vehicle_at(x, y):
            self.toast("A car is standing there")
        elif value in (DEST, SPAWN) and lane == ANY:
            self.toast("Not inside a junction")
        else:
            c.grid[y, x] = JUNCTION if (value == ROAD and lane == ANY) else value
            c.validate_trips()

    def right_click(self, cell):
        c = self.city
        v = c.vehicle_at(*cell)
        if v:
            self.selected = v.id
            return
        sel = c.get_vehicle(self.selected)
        if sel and c.is_walkable(*cell):
            path = c.route((sel.x, sel.y), cell)
            if path is None:
                self.toast("No route to that cell")
            else:
                sel.dest, sel.custom_dest, sel.path = cell, True, path

    def update(self, dt):
        if not self.paused:
            self.acc += dt
            while self.acc >= self.step_ms:
                self.city.step(self.auto)
                self.acc -= self.step_ms
        k = 1 - math.exp(-dt / 70)          # ease the drawn position towards the real cell
        for v in self.city.vehicles:
            v.rx += (v.x - v.rx) * k
            v.ry += (v.y - v.ry) * k

    # ---- drawing ----
    def draw(self):
        s, c = self.screen, self.city
        s.fill(BG)
        pygame.draw.rect(s, PANEL, (MARGIN - 6, MARGIN - 6, self.gw + 12, self.gh + 12), border_radius=14)
        active = {v.dest for v in c.vehicles}
        for y in range(ROWS):
            for x in range(COLS):
                r, t = self.cell_rect(x, y), c.grid[y, x]
                if t == BUILDING:
                    pygame.draw.rect(s, ROAD_A, r)
                    self.draw_building(r, x, y)
                    continue
                pygame.draw.rect(s, JUNC_C if c.lane[y, x] == ANY else (ROAD_A if (x + y) % 2 == 0 else ROAD_B), r)
                self.draw_road_marks(r, x, y)
                if t == DEST:
                    self.draw_dest(r, (x, y) in active)
                elif t == SPAWN:
                    self.draw_spawn(r, int(c.lane[y, x]))
                elif t == BLOCK:
                    self.draw_block(r)
        if self.grid_lines:
            for x in range(COLS + 1):
                pygame.draw.line(s, GRID_LINE, (MARGIN + x * CELL, MARGIN), (MARGIN + x * CELL, MARGIN + self.gh))
            for y in range(ROWS + 1):
                pygame.draw.line(s, GRID_LINE, (MARGIN, MARGIN + y * CELL), (MARGIN + self.gw, MARGIN + y * CELL))
        self.draw_lights()
        if self.show_paths:
            self.draw_paths()
        for v in c.vehicles:
            self.draw_vehicle(v, v.id == self.selected)
        cell = self.mouse_cell(pygame.mouse.get_pos())
        if cell:
            pygame.draw.rect(s, TOOL_COLORS[self.tool], self.cell_rect(*cell), 2, border_radius=6)
        self.draw_sidebar()
        self.draw_footer(cell)
        pygame.display.flip()

    def draw_building(self, r, x, y):
        body_c, top_c = BLD_TONES[(x * 3 + y * 5) % len(BLD_TONES)]
        body = r.inflate(-6, -6)
        pygame.draw.rect(self.screen, BLD_SH, body.move(2, 3), border_radius=6)     # shadow
        pygame.draw.rect(self.screen, body_c, body, border_radius=6)
        pygame.draw.rect(self.screen, top_c, (body.x + 3, body.y + 3, body.w - 6, 4), border_radius=2)
        for i in (0, 1):                                                            # windows
            for j in (0, 1):
                pygame.draw.rect(self.screen, WINDOW_C, (body.x + 7 + i * 11, body.y + 13 + j * 8, 6, 4))

    def draw_road_marks(self, r, x, y):
        s, c = self.screen, self.city
        code = int(c.lane[y, x])
        if code not in DIRS:
            return
        dx, dy = DIRS[code]
        for px, py in ((dy, dx), (-dy, -dx)):       # dashed centre line next to the opposite lane
            nx, ny = x + px, y + py
            if c.in_bounds(nx, ny) and c.lane[ny, nx] == OPPOSITE[code]:
                if py:
                    ye = r.bottom - 1 if py > 0 else r.top
                    pygame.draw.line(s, LANE_LINE, (r.left + 4, ye), (r.left + 15, ye), 2)
                    pygame.draw.line(s, LANE_LINE, (r.left + 21, ye), (r.right - 4, ye), 2)
                else:
                    xe = r.right - 1 if px > 0 else r.left
                    pygame.draw.line(s, LANE_LINE, (xe, r.top + 4), (xe, r.top + 15), 2)
                    pygame.draw.line(s, LANE_LINE, (xe, r.top + 21), (xe, r.bottom - 4), 2)
        if (x + y) % 2 == 0:                        # faint direction arrow
            cx, cy = r.center
            pygame.draw.polygon(s, ARROW_C, [(cx + dx * 6, cy + dy * 6),
                                             (cx - dx * 4 + dy * 5, cy - dy * 4 + dx * 5),
                                             (cx - dx * 4 - dy * 5, cy - dy * 4 - dx * 5)])

    def draw_dest(self, r, has_trip):
        b = r.inflate(-8, -8)
        pygame.draw.rect(self.screen, (24, 66, 58) if has_trip else (34, 50, 52), b, border_radius=6)
        pygame.draw.rect(self.screen, GREEN, b, 2, border_radius=6)
        self.text("P", r.center, self.f_sm, GREEN, "center")

    def draw_spawn(self, r, code):
        dx, dy = DIRS.get(code, (1, 0))
        cx, cy = r.center
        pygame.draw.circle(self.screen, YELLOW, (cx, cy), 12, 2)
        pygame.draw.polygon(self.screen, YELLOW, [(cx + dx * 6, cy + dy * 6),
                                                  (cx - dx * 3 + dy * 5, cy - dy * 3 + dx * 5),
                                                  (cx - dx * 3 - dy * 5, cy - dy * 3 - dx * 5)])

    def draw_block(self, r):
        b = r.inflate(-4, -4)
        pygame.draw.rect(self.screen, (60, 36, 12), b, border_radius=4)
        self.screen.set_clip(b)                     # orange hazard stripes
        for i in range(-1, 4):
            pygame.draw.line(self.screen, ORANGE, (b.x + i * 12, b.bottom), (b.x + i * 12 + 12, b.y), 4)
        self.screen.set_clip(None)
        pygame.draw.rect(self.screen, ORANGE, b, 2, border_radius=4)

    def draw_lights(self):
        for L in self.city.lights:
            for ax, ay, code in L.approaches:
                r = self.cell_rect(ax, ay)
                bar = {EAST: pygame.Rect(r.right - 4, r.top + 2, 4, r.h - 4),
                       WEST: pygame.Rect(r.left, r.top + 2, 4, r.h - 4),
                       SOUTH: pygame.Rect(r.left + 2, r.bottom - 4, r.w - 4, 4),
                       NORTH: pygame.Rect(r.left + 2, r.top, r.w - 4, 4)}[code]
                if L.allows(code):
                    color = GREEN
                elif L.yellow and ((code in (SOUTH, NORTH)) == (L.phase == 0)):
                    color = YELLOW                  # about to turn red
                else:
                    color = RED
                pygame.draw.rect(self.screen, color, bar, border_radius=2)

    def draw_paths(self):
        overlay = pygame.Surface((self.gw, self.gh), pygame.SRCALPHA)
        for v in self.city.vehicles:
            if not v.path:
                continue
            pts = [(v.rx * CELL + CELL / 2, v.ry * CELL + CELL / 2)]
            pts += [(x * CELL + CELL / 2, y * CELL + CELL / 2) for x, y in v.path]
            sel = v.id == self.selected
            pygame.draw.lines(overlay, (*v.color, 170 if sel else 70), False, pts, 4 if sel else 2)
            if sel:
                pygame.draw.circle(overlay, (*v.color, 220), pts[-1], 7, 2)
        self.screen.blit(overlay, (MARGIN, MARGIN))

    def draw_vehicle(self, v, selected):
        s = self.screen
        cx, cy = int(MARGIN + v.rx * CELL + CELL / 2), int(MARGIN + v.ry * CELL + CELL / 2)
        pygame.draw.ellipse(s, (8, 9, 13), (cx - 12, cy + 5, 24, 9))               # shadow
        horiz = v.hx != 0
        body = pygame.Rect(0, 0, 26 if horiz else 15, 15 if horiz else 26)
        body.center = (cx, cy)
        pygame.draw.rect(s, v.color, body, border_radius=5)
        pygame.draw.rect(s, tuple(min(255, ch + 60) for ch in v.color), body, 2, border_radius=5)
        glass = pygame.Rect(0, 0, 6 if horiz else 11, 11 if horiz else 6)         # windscreen
        glass.center = (cx + v.hx * 6, cy + v.hy * 6)
        pygame.draw.rect(s, (20, 24, 36), glass, border_radius=2)
        if v.wait > 0:                                                             # brake light
            pygame.draw.circle(s, RED, (cx - v.hx * 11, cy - v.hy * 11), 3)
        if selected:
            radius = int(19 + 2 * math.sin(pygame.time.get_ticks() / 180))
            pygame.draw.circle(s, (255, 255, 255), (cx, cy), radius, 2)

    def draw_sidebar(self):
        s, c = self.screen, self.city
        px = 2 * MARGIN + self.gw
        panel = pygame.Rect(px, MARGIN - 6, SIDEBAR - 10, self.H - 2 * (MARGIN - 6))
        pygame.draw.rect(s, PANEL, panel, border_radius=14)
        x0, x1 = px + 18, panel.right - 18
        self.text("City Traffic Sandbox", (x0, 24), self.f_lg)
        self.text("multi-agent signal control", (x0, 54), self.f_md, MUTED)

        pills = [("PAUSED" if self.paused else "RUNNING", not self.paused),
                 ("SPAWN ON" if self.auto else "SPAWN OFF", self.auto)]
        for i, (label, on) in enumerate(pills):
            pill = pygame.Rect(x0 + i * 106, 88, 98, 26)
            color = GREEN if on else MUTED
            pygame.draw.rect(s, (30, 34, 48), pill, border_radius=13)
            pygame.draw.rect(s, color, pill, 2, border_radius=13)
            self.text(label, pill.center, self.f_sm, color, "center")

        self.text("STATS", (x0, 134), self.f_sm, ACCENT)
        avg = c.arrived_wait / c.arrived if c.arrived else 0.0
        stats = [("Tick", c.tick), ("Vehicles", len(c.vehicles)), ("Arrived", c.arrived),
                 ("Avg wait / trip", f"{avg:.1f}"), ("Stopped now", sum(v.wait > 0 for v in c.vehicles)),
                 ("Signals", POLICIES[self.policy_idx][0]), ("Spawn rate", f"{c.spawn_rate:.2f}"),
                 ("Speed (steps/s)", f"{1000 / self.step_ms:.1f}")]
        for i, (name, value) in enumerate(stats):
            self.text(name, (x0, 158 + i * 22), self.f_md, MUTED)
            self.text(str(value), (x1, 158 + i * 22), self.f_md, TEXT, "topright")

        self.text("TOOLS  (keys 1-4)", (x0, 346), self.f_sm, ACCENT)
        for i, (name, _) in enumerate(TOOLS):
            y = 370 + i * 26
            if i == self.tool:
                pygame.draw.rect(s, (40, 45, 66), (x0 - 8, y - 4, x1 - x0 + 16, 26), border_radius=8)
            pygame.draw.rect(s, TOOL_COLORS[i], (x0, y + 2, 14, 14), border_radius=4)
            self.text(str(i + 1), (x0 + 26, y), self.f_md, MUTED)
            self.text(name, (x0 + 46, y), self.f_md, TEXT if i == self.tool else MUTED)

        self.text("KEYS", (x0, 486), self.f_sm, ACCENT)
        keys = [("Space", "run / pause"), ("D", "auto-spawn traffic"), ("T", "spawn a car"),
                ("N / X", "add / remove car"), ("M", "signal mode"), ("R / C", "new layout / clear"),
                ("S / L", "save / load layout"), ("G / P", "grid lines / paths"),
                ("+ / -", "faster / slower"), ("[ / ]", "spawn rate")]
        for i, (key, desc) in enumerate(keys):
            self.text(key, (x0, 508 + i * 18), self.f_sm, TEXT)
            self.text(desc, (x0 + 62, 508 + i * 18), self.f_sm, MUTED)

    def draw_footer(self, cell):
        y = MARGIN + self.gh + 20
        self.text("Left-drag: paint roads   |   Right-click car: select   |   Right-click cell: re-route it",
                  (MARGIN, y), self.f_md, MUTED)
        if pygame.time.get_ticks() < self.toast_until:
            self.text(self.toast_text, (MARGIN, y + 28), self.f_md, ACCENT)
        elif cell:
            g, lane = int(self.city.grid[cell[1], cell[0]]), int(self.city.lane[cell[1], cell[0]])
            extra = f" [{LANE_NAMES[lane]}]" if lane else ""
            self.text(f"Cell ({cell[0]}, {cell[1]}) - {CELL_NAMES[g]}{extra}", (MARGIN, y + 28))
        v = self.city.get_vehicle(self.selected)
        if v:
            self.text(f"Car {v.id}: heading to {v.dest}, waited {v.total_wait} ticks",
                      (MARGIN, y + 52), self.f_md, v.color)

    def run(self):
        while self.running:
            dt = self.clock.tick(60)
            for e in pygame.event.get():
                self.handle_event(e)
            self.update(dt)
            self.draw()
        pygame.quit()


if __name__ == "__main__":
    App().run()