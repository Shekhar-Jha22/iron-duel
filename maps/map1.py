from .base import MapConfig, Spawn

MAP1 = MapConfig(
    name="Combat Arena",
    width=800,
    height=500,
    player_spawn=Spawn(x=160, y=250),
    enemy_spawn=Spawn(x=640, y=250),
    obstacles=[]
)
