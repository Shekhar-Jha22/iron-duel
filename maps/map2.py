from .base import MapConfig, Spawn, Obstacle

MAP2 = MapConfig(
    name="Symmetric Arena",
    width=1200,
    height=1200,
    player_spawn=Spawn(x=100, y=600),
    enemy_spawn=Spawn(x=1100, y=600),
    obstacles=[
        # Central square: center (600,600), size 200x200
        Obstacle(x=500, y=500, w=200, h=200, type="wall_chunk"),
        # Corner square top-left: top-left at (200,200), size 100x100
        Obstacle(x=200, y=200, w=100, h=100, type="boulder"),
        # Corner square bottom-right: top-left at (1000,1000), size 100x100
        Obstacle(x=1000, y=1000, w=100, h=100, type="boulder"),
        # Vertical rectangle left side: center (200,850), corners (150,700)-(250,1000)
        Obstacle(x=150, y=700, w=100, h=300, type="wall_chunk"),
        # Vertical rectangle right side: center (1000,350), corners (950,200)-(1050,500)
        Obstacle(x=950, y=200, w=100, h=300, type="wall_chunk"),
    ],
    max_speed=15.0,
    accel=3,
)