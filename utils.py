import pygame
import math

FPS = 30
BATTLE_DURATION = 30 * FPS
WIDTH = 400

def get_state(robot, enemy):
    dx = enemy.x - robot.x
    dy = enemy.y - robot.y
    dist = math.hypot(dx, dy) / WIDTH
    angle_to_enemy = (math.degrees(math.atan2(dy, dx)) - robot.angle) % 360 / 360
    return [
        dist,
        angle_to_enemy,
        robot.hp / 3,
        enemy.hp / 3,
        int(robot.walls["left"]["active"]),
        int(robot.walls["right"]["active"]),
        int(robot.walls["back"]["active"]),
        int(enemy.walls["left"]["active"]),
        int(enemy.walls["right"]["active"]),
        int(enemy.walls["back"]["active"]),
    ]

def check_hit(attacker, defender):
    fx, fy = attacker.get_front_tip()
    wall_zones = defender.get_wall_zones()

    hit_reward = 0
    penalty = 0

    for side, (wx, wy) in wall_zones.items():
        if defender.walls[side]["active"]:
            dist = ((fx - wx) ** 2 + (fy - wy) ** 2) ** 0.5
            if dist < 10:  # чувствительность попадания
                defender.walls[side]["active"] = False
                defender.hp -= 1
                hit_reward += 2
                penalty -= 1
                break

    return hit_reward, penalty
    return 0, 0


def draw_ui(win, timer, robot1, robot2):
    font = pygame.font.SysFont("Arial", 18)
    time_s = max(0, (BATTLE_DURATION - timer) // FPS)

    text = f"Time: {time_s}s  |  R1 HP: {robot1.hp} Score: {robot1.score:.1f}  |  R2 HP: {robot2.hp} Score: {robot2.score:.1f}"
    surf = font.render(text, True, (0, 0, 0))
    win.blit(surf, (10, 10))  # Верхняя часть окна