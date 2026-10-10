import json
import os
import numpy as np
from flask import Flask, g, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sock import Sock
from simple_websocket import ConnectionClosed

from env import MiniMilitiaEnv
from maps import MAPS
from agents import (
    RandomAgent, 
    PPOAgent,
    ChaserAgent, 
    CirclerAgent,
    DefensiveHiderAgent, 
    StationaryAgent,
    MobileDefenseAgent,
    MAP1_AGENTS,
    MAP2_AGENTS,
    STAGE_NAMES,
)
from agents.numpy_policy import NumpyPolicy
from observations import OBS_DIM
from rewards import REWARDS
from auth import login_required, register_auth_routes, verify_token
from matches import TICK_S, Match, MatchManager, match_winner

app = Flask(__name__)
CORS(app)
register_auth_routes(app)
sock = Sock(app)
manager = MatchManager()

AGENT_REGISTRY = {
    "map1": {
        "chaser": ChaserAgent,
        "circler": CirclerAgent,
    },
    "map2": {
        "hider": DefensiveHiderAgent,
        "mobile": MobileDefenseAgent,
    },
}
PPO_VARIATION = os.environ.get("PPO_VARIATION", "v1")
_model_cache = {}


def _effective_name(name, map_id):
    """Name of the agent that will actually play (unknown names fall back to random)."""
    if name in ("human", "ppo", "random") or name in AGENT_REGISTRY.get(map_id, {}):
        return name
    return "random"


def _ppo_checkpoint(map_id, opponent):
    """Stage checkpoint trained against `opponent`; self-play checkpoint for human/ppo/unknown."""
    d, v = f"models/{map_id}", PPO_VARIATION
    stages = STAGE_NAMES[map_id]
    selfplay = [f"{d}/final_{v}.npz", f"{d}/{v}_stage_{len(stages) - 1}_final.npz"]
    cands = []
    # "random" is a ~0-2% warm-up stage, so its checkpoint is undertrained → use self-play
    if opponent in stages and opponent not in ("selfplay", "random"):
        cands.append(f"{d}/{v}_stage_{stages.index(opponent)}_final.npz")
    cands += selfplay + [f"{d}/{v}_best.npz", f"{d}/best.npz", f"{d}/ppo.npz"]
    return next((p for p in cands if os.path.exists(p) and _policy(p) is not None), None)


def _policy(path):
    """Load (and cache) a torch-free policy; None if it was trained on a different obs layout."""
    if path not in _model_cache:
        pol = NumpyPolicy(path)
        if pol.obs_dim != OBS_DIM:
            print(f"⚠ skipping {path}: obs {pol.obs_dim} ≠ current {OBS_DIM} (retrain)")
            pol = None
        _model_cache[path] = pol
    return _model_cache[path]


def _load_ppo(map_id, opponent, team):
    agent = PPOAgent(team=team)
    path = _ppo_checkpoint(map_id, _effective_name(opponent, map_id))
    if path is None:
        print(f"⚠ No PPO checkpoint for {map_id} (run tools/export_policies.py)")
        return agent
    agent.model = _policy(path)
    print(f"✓ PPO {map_id} vs {opponent}: {path}")
    return agent


def _make_agent(name, map_id="map1", team=0, opponent=None):
    if name == "random":
        return RandomAgent(team=team)
    if name == "ppo":
        return _load_ppo(map_id, opponent, team)
    agent_cls = AGENT_REGISTRY.get(map_id, {}).get(name, RandomAgent)
    return agent_cls(team=team)

@app.route("/api/agents", methods=["GET"])
def list_agents():
    return jsonify({
        "agents": [
            {"id": "random", "name": "Random", "color": "#ef4444", "ready": True},
            {"id": "rulebased", "name": "Rule-Based", "color": "#f59e0b", "ready": True},
            {"id": "ppo", "name": "PPO Agent", "color": "#22c55e", "ready": True},
        ]
    })

@app.route("/api/maps", methods=["GET"])
def list_maps():
    return jsonify({
        "maps": [
            {"id": m_id, "name": m.name}
            for m_id, m in MAPS.items()
        ]
    })

def _valid_agents(map_id):
    return {"random", "ppo", *AGENT_REGISTRY.get(map_id, {})}


def _owned_match(match_id):
    m = manager.get(match_id)
    if m is None:
        return None, None, (jsonify({"error": "Match not found."}), 404)
    seat = m.seat_of(g.user["id"])
    if not m.can_view(g.user["id"]):
        return None, None, (jsonify({"error": "Not your match."}), 403)
    return m, seat, None


@app.post("/api/match/start")
@login_required
def start_match():
    data = request.get_json(silent=True) or {}
    mode, map_id = data.get("mode"), data.get("map_id", "map1")
    if map_id not in MAPS:
        return jsonify({"error": "Unknown map."}), 400
    valid = _valid_agents(map_id)
    user = g.user
    if mode in ("bot", "rl"):
        opp = "ppo" if mode == "rl" else data.get("opponent", "random")
        if opp not in valid:
            return jsonify({"error": "Unknown opponent."}), 400
        match = Match(mode, map_id, seats=[user["id"], None],
                      agents=[None, _make_agent(opp, map_id, team=1, opponent="human")],
                      labels=["human", opp], names=[user["name"], opp])
        seat = 0
    elif mode == "sim":
        a0, a1 = data.get("agent0", "ppo"), data.get("agent1", "random")
        if a0 not in valid or a1 not in valid:
            return jsonify({"error": "Unknown agent."}), 400
        tick_ms = min(max(int(data.get("tick_ms", 80)), 30), 200)
        match = Match(mode, map_id, seats=[None, None],
                      agents=[_make_agent(a0, map_id, team=0, opponent=a1),
                              _make_agent(a1, map_id, team=1, opponent=a0)],
                      labels=[a0, a1], names=[a0, a1], owner=user["id"], tick_s=tick_ms / 1000)
        seat = None
    else:
        return jsonify({"error": "Unknown mode."}), 400
    manager.add(match)
    return jsonify(match.snapshot(seat))


@sock.route("/api/match/<match_id>/ws")
def match_ws(ws, match_id):
    """Push a snapshot every tick; receive {"action": [...]}, {"ping": 1} or {"tick_ms": n}.
    The first client message must be {"token": ...} so credentials never appear in URLs."""
    try:
        user = verify_token(json.loads(ws.receive(timeout=10) or "{}").get("token"))
    except (ValueError, AttributeError, ConnectionClosed):
        return
    m = manager.get(match_id)
    if user is None or m is None or not m.can_view(user["id"]):
        err = "Please log in again." if user is None else "Match not found." if m is None else "Not your match."
        ws.send(json.dumps({"error": err}))
        return
    seat = m.seat_of(user["id"])
    seen = -1
    try:
        while True:
            while (raw := ws.receive(timeout=0)) is not None:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if seat is not None and "action" in msg:
                    m.set_input(seat, msg["action"])
                elif m.kind == "sim" and "tick_ms" in msg:
                    m.tick_s = min(max(int(msg["tick_ms"]), 30), 200) / 1000
                m.heartbeat(seat)
            version = m.wait_for_update(seen, timeout=TICK_S * 2)
            if version != seen:
                seen = version
                snap = m.snapshot(seat)
                ws.send(json.dumps(snap, separators=(",", ":")))
                if snap["done"]:
                    break
    except ConnectionClosed:
        pass


@app.post("/api/match/<match_id>/leave")
@login_required
def leave_match(match_id):
    m, seat, err = _owned_match(match_id)
    if err:
        return err
    if m.kind == "pvp" and seat is not None and not m.done:
        m.forfeit(seat, "left")
    else:
        manager.remove(m.id)
    return jsonify({"status": "left"})


@app.post("/api/queue/join")
@login_required
def queue_join():
    map_id = (request.get_json(silent=True) or {}).get("map_id", "map1")
    if map_id not in MAPS:
        return jsonify({"error": "Unknown map."}), 400
    return jsonify(manager.join_queue(g.user, map_id))


@app.get("/api/queue/status")
@login_required
def queue_status():
    return jsonify(manager.queue_status(g.user))


@app.post("/api/queue/leave")
@login_required
def queue_leave():
    manager.leave_queue(g.user)
    return jsonify({"status": "left"})

@app.route("/api/benchmark", methods=["POST"])
@login_required
def benchmark():
    data = request.json or {}
    a0_name = data.get("agent0", "ppo")
    a1_name = data.get("agent1", "rulebased")
    n_episodes = max(1, min(int(data.get("episodes", 20)), int(os.environ.get("BENCH_MAX_EPISODES", 200))))
    map_id = data.get("map_id", "map1")

    map_cfg = MAPS[map_id]
    reward_cls = REWARDS[map_id]

    results = {"agent0_wins": 0, "agent1_wins": 0, "draws": 0, "avg_steps": 0, "episodes": n_episodes}
    total_steps = 0

    for _ in range(n_episodes):
        env = MiniMilitiaEnv(map_cfg=map_cfg, reward_fn=reward_cls())
        obs0, _ = env.reset()
        obs1 = env._get_obs(1)
        agents = [_make_agent(a0_name, map_id=map_id, team=0, opponent=a1_name),
                  _make_agent(a1_name, map_id=map_id, team=1, opponent=a0_name)]
        done = False

        while not done:
            st = env.get_full_state()
            act0 = agents[0].get_action(obs0, st)
            act1 = agents[1].get_action(obs1, st)
            obs0, _, term, trunc, info = env.step([act0, act1])
            obs1 = env._get_obs(1)
            done = term or trunc

        total_steps += info["step"]
        w = match_winner(info, a0_name, a1_name)
        results["agent0_wins" if w == 0 else "agent1_wins" if w == 1 else "draws"] += 1

    results["avg_steps"] = round(total_steps / n_episodes, 1)
    results["agent0_winrate"] = round(results["agent0_wins"] / n_episodes * 100, 1)
    results["agent1_winrate"] = round(results["agent1_wins"] / n_episodes * 100, 1)
    results["agent0_name"] = a0_name
    results["agent1_name"] = a1_name
    return jsonify(results)

DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")


@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def frontend(path):
    if not os.path.isdir(DIST):
        return jsonify({"error": "Frontend not built. Run `npm run build` in frontend/."}), 404
    if path and os.path.isfile(os.path.join(DIST, path)):
        return send_from_directory(DIST, path)
    return send_from_directory(DIST, "index.html")


if __name__ == "__main__":
    print("\n Mini Militia RL — API Server")
    print(" Running on http://localhost:5000\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)