import threading
import time
import uuid

import numpy as np

from env import MiniMilitiaEnv
from maps import MAPS
from rewards import REWARDS

TICK_S = 0.08              # normal game speed (12.5 ticks/s)
PVP_COUNTDOWN_S = 3
INPUT_TIMEOUT_S = 12       # human seat silent this long → forfeit
VIEWER_TIMEOUT_S = 12      # simulation nobody is watching → stop
QUEUE_TIMEOUT_S = 30       # waiting player who stopped polling is dropped (hidden tabs poll slowly)
MATCH_TTL_S = 15 * 60      # finished/abandoned matches are purged after this
MAX_CATCHUP = 3            # ticks to replay at once after a CPU stall
IDLE_ACTION = np.zeros(3, dtype=np.int64)


def sanitize_action(raw):
    try:
        move, aim, shoot = (int(v) for v in raw)
    except (TypeError, ValueError):
        return IDLE_ACTION.copy()
    return np.array([min(max(move, 0), 8), min(max(aim, 0), 15), 1 if shoot else 0], dtype=np.int64)


def match_winner(info, a0_name, a1_name):
    """0 / 1 / None (draw). A timeout with both alive counts as a loss for a PPO side."""
    if info["p0_alive"] and not info["p1_alive"]:
        return 0
    if info["p1_alive"] and not info["p0_alive"]:
        return 1
    ppo0, ppo1 = a0_name == "ppo", a1_name == "ppo"
    if ppo0 != ppo1:
        return 1 if ppo0 else 0
    if ppo0 and ppo1:
        return None
    if info["p0_health"] == info["p1_health"]:
        return None
    return 0 if info["p0_health"] > info["p1_health"] else 1


class Match:
    """seats[i] = user id of a human seat or None for an AI seat; agents[i] = AI agent or None.

    Every match advances on the server clock; clients only send inputs and receive snapshots.
    """

    def __init__(self, kind, map_id, seats, agents, labels, names, owner=None, tick_s=TICK_S):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.map_id = map_id
        self.seats = seats
        self.agents = agents
        self.labels = labels
        self.names = names
        self.owner = owner
        self.tick_s = tick_s
        self.env = MiniMilitiaEnv(map_cfg=MAPS[map_id], reward_fn=REWARDS[map_id]())
        self.env.reset()
        now = time.time()
        self.last_input = [now, now]
        self.viewer_seen = now
        self.touched = now
        self.start_at = now + (PVP_COUNTDOWN_S if kind == "pvp" else 0.6)
        self.next_tick = self.start_at
        self.inputs = [IDLE_ACTION.copy(), IDLE_ACTION.copy()]
        self.done = False
        self.result = None
        self.version = 0
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)

    def seat_of(self, uid):
        return self.seats.index(uid) if uid in self.seats else None

    def can_view(self, uid):
        return uid in self.seats or uid == self.owner

    def _finish(self, winner, steps, reason):
        self.done = True
        self.result = {"winner": winner, "steps": steps, "reason": reason,
                       "agent0": self.labels[0], "agent1": self.labels[1], "map_id": self.map_id}

    def set_input(self, seat, raw):
        with self.lock:
            self.inputs[seat] = sanitize_action(raw)
            self.last_input[seat] = time.time()

    def heartbeat(self, seat):
        now = time.time()
        self.touched = now
        self.viewer_seen = now
        if seat is not None:
            self.last_input[seat] = now

    def _step_locked(self):
        state = self.env.get_full_state()
        actions = []
        for i, agent in enumerate(self.agents):
            if agent is None:
                actions.append(self.inputs[i])
                continue
            try:
                obs = self.env._get_obs(i) if getattr(agent, "uses_obs", False) else None
                actions.append(agent.get_action(obs, state))
            except Exception as e:
                print(f"ERROR: agent {i} in match {self.id} failed: {e}")
                actions.append(IDLE_ACTION.copy())
        _, _, terminated, truncated, info = self.env.step(actions)
        if terminated or truncated:
            self._finish(match_winner(info, *self.labels), info["step"], "combat")

    def advance(self, now):
        """Run every tick that is due (bounded catch-up), then wake subscribers."""
        with self.lock:
            if self.done or now < self.next_tick:
                return
            for seat, uid in enumerate(self.seats):
                if uid is not None and now - self.last_input[seat] > INPUT_TIMEOUT_S:
                    self._finish(1 - seat, self.env.step_count, "disconnect")
            if not self.done and self.kind == "sim" and now - self.viewer_seen > VIEWER_TIMEOUT_S:
                self._finish(None, self.env.step_count, "abandoned")
            steps = 0
            while not self.done and now >= self.next_tick and steps < MAX_CATCHUP:
                self._step_locked()
                self.next_tick += self.tick_s
                steps += 1
            if now >= self.next_tick:
                self.next_tick = now + self.tick_s
            self.version += 1
            self.changed.notify_all()

    def forfeit(self, seat, reason):
        with self.lock:
            if not self.done:
                self._finish(1 - seat, self.env.step_count, reason)
                self.version += 1
                self.changed.notify_all()

    def wait_for_update(self, seen_version, timeout):
        with self.lock:
            if self.version == seen_version and not self.done:
                self.changed.wait(timeout)
            return self.version

    def snapshot(self, seat=None):
        with self.lock:
            return {
                "match_id": self.id,
                "kind": self.kind,
                "map_id": self.map_id,
                "seat": seat,
                "players": self.names,
                "tick_ms": round(self.tick_s * 1000),
                "state": self.env.get_full_state(),
                "done": self.done,
                "starts_in": max(0.0, round(self.start_at - time.time(), 1)),
                "match_info": self.result,
            }


class MatchManager:
    def __init__(self):
        self.matches = {}
        self.waiting = {}   # uid → {"map_id", "name", "ts", "match_id", "seat"}
        self.lock = threading.Lock()
        threading.Thread(target=self._tick_loop, daemon=True).start()

    # ── matches ───────────────────────────────────────────────────────────
    def add(self, match):
        with self.lock:
            self.matches[match.id] = match
        return match

    def get(self, match_id):
        with self.lock:
            m = self.matches.get(match_id)
        if m is not None:
            m.touched = time.time()
        return m

    def remove(self, match_id):
        with self.lock:
            m = self.matches.pop(match_id, None)
        if m is not None:
            with m.lock:
                m.done = True
                m.changed.notify_all()

    def _tick_loop(self):
        last_purge = 0.0
        while True:
            now = time.time()
            with self.lock:
                live = [m for m in self.matches.values() if not m.done]
            for m in live:
                m.advance(now)
            if now - last_purge > 2:
                self._purge(now)
                last_purge = now
            due = min((m.next_tick for m in live if not m.done), default=now + 0.05)
            time.sleep(min(0.05, max(0.004, due - time.time())))

    def _purge(self, now):
        with self.lock:
            for mid in [mid for mid, m in self.matches.items() if now - m.touched > MATCH_TTL_S]:
                del self.matches[mid]
            for uid in [u for u, w in self.waiting.items() if self._stale(w, now)]:
                del self.waiting[uid]

    def _stale(self, w, now):
        if w["match_id"] is None:
            return now - w["ts"] > QUEUE_TIMEOUT_S
        m = self.matches.get(w["match_id"])
        return m is None or m.done

    # ── matchmaking ───────────────────────────────────────────────────────
    # A matched entry stays in `waiting` until its match ends, so join/status keep
    # returning the match to both players even if one response was lost.
    def join_queue(self, user, map_id):
        uid = user["id"]
        with self.lock:
            mine = self.waiting.get(uid)
            if mine and mine["match_id"]:
                if mine["map_id"] == map_id and not self._stale(mine, time.time()):
                    return self._matched(mine)
                del self.waiting[uid]
            opponent = next((u for u, w in self.waiting.items()
                             if u != uid and w["map_id"] == map_id and w["match_id"] is None), None)
            if opponent is None:
                self.waiting[uid] = {"map_id": map_id, "name": user["name"], "ts": time.time(),
                                     "match_id": None, "seat": None}
                return {"status": "waiting"}
            opp = self.waiting[opponent]
            match = Match("pvp", map_id, seats=[opponent, uid], agents=[None, None],
                          labels=["human", "human"], names=[opp["name"], user["name"]])
            self.matches[match.id] = match
            opp.update(match_id=match.id, seat=0, ts=time.time())
            self.waiting[uid] = {"map_id": map_id, "name": user["name"], "ts": time.time(),
                                 "match_id": match.id, "seat": 1}
            return self._matched(self.waiting[uid])

    def queue_status(self, user):
        uid = user["id"]
        with self.lock:
            mine = self.waiting.get(uid)
            if mine is None or self._stale(mine, time.time()):
                self.waiting.pop(uid, None)
                return {"status": "idle"}
            if mine["match_id"]:
                return self._matched(mine)
            mine["ts"] = time.time()
            return {"status": "waiting",
                    "players_waiting": sum(1 for w in self.waiting.values() if w["match_id"] is None)}

    def _matched(self, w):
        m = self.matches.get(w["match_id"])
        if m is not None:
            m.last_input[w["seat"]] = time.time()
        return {"status": "matched", "match_id": w["match_id"], "seat": w["seat"]}

    def leave_queue(self, user):
        with self.lock:
            w = self.waiting.get(user["id"])
            if w is not None and w["match_id"] is None:
                del self.waiting[user["id"]]
