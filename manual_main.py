import pygame
import math
import torch
import torch.nn as nn
from robot import Robot
from utils import get_state, check_hit
from agent import Agent

# Нейросеть
class Net(nn.Module):
    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Настройки
pygame.init()
WIDTH, HEIGHT = 400, 400
FPS = 60
UI_HEIGHT = 50
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Manual vs AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 18)

def main():
    # Загрузка модели
    network = Net()
    agent = Agent(network)
    agent.load("runs/1/best_model_round_39.pt")
    agent.net.eval()

    MARGIN = 40
    arena_top = UI_HEIGHT
    arena_height = HEIGHT - UI_HEIGHT

    robot1 = Robot(MARGIN, arena_top + MARGIN, (0, 120, 255), "R1")
    robot2 = Robot(WIDTH - MARGIN, arena_top + arena_height - MARGIN, (255, 60, 60), "R2", agent)

    BATTLE_DURATION = 30 * FPS
    timer = 0
    run = True

    while run:
        clock.tick(FPS)
        timer += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Управление человеком
        keys = pygame.key.get_pressed()
        old_x, old_y, old_angle = robot1.x, robot1.y, robot1.angle

        if keys[pygame.K_w]:
            robot1.x += math.cos(math.radians(robot1.angle)) * 2
            robot1.y += math.sin(math.radians(robot1.angle)) * 2
        if keys[pygame.K_s]:
            robot1.x -= math.cos(math.radians(robot1.angle)) * 2
            robot1.y -= math.sin(math.radians(robot1.angle)) * 2
        if keys[pygame.K_a]:
            robot1.angle -= 5
        if keys[pygame.K_d]:
            robot1.angle += 5

        robot1.x = max(robot1.size / 2, min(WIDTH - robot1.size / 2, robot1.x))
        robot1.y = max(UI_HEIGHT + robot1.size / 2, min(HEIGHT - robot1.size / 2, robot1.y))

        if robot1.collides_with(robot2):
            robot1.x, robot1.y, robot1.angle = old_x, old_y, old_angle

        # AI
        if robot2.agent:
            state2 = get_state(robot2, robot1)
            action2 = robot2.agent.select_action(state2)
        else:
            action2 = 0
        robot2.move(action2, robot1, WIDTH, HEIGHT)

        # === Счёт для ИИ
        if action2 in [1, 2]: robot2.score += 0.1
        if action2 in [3, 4]: robot2.score -= 0.2
        if robot2.collides_with(robot1): robot2.score -= 0.5

        def in_corner(robot):
            margin = 30
            return (
                (robot.x < margin and robot.y < margin) or
                (robot.x > WIDTH - margin and robot.y < margin) or
                (robot.x < margin and robot.y > HEIGHT - margin) or
                (robot.x > WIDTH - margin and robot.y > HEIGHT - margin)
            )

        if in_corner(robot2): robot2.score -= 0.5

        # Удары и очки
        _, r2_penalty = check_hit(robot1, robot2)
        r2_reward, _ = check_hit(robot2, robot1)
        robot2.score += r2_reward + r2_penalty

        # Перезапуск
        if timer >= BATTLE_DURATION or robot1.hp <= 0 or robot2.hp <= 0:
            robot1.reset()
            robot2.reset()
            timer = 0

        # === Отрисовка
        win.fill((240, 240, 240))
        pygame.draw.rect(win, (150, 150, 150), (0, UI_HEIGHT, WIDTH, HEIGHT - UI_HEIGHT), 5)
        robot1.draw(win)
        robot2.draw(win)

        # UI
        text = font.render(
            f"Time: {timer // FPS}s | R1 HP: {robot1.hp} | R2 HP: {robot2.hp} | R2 Score: {robot2.score:.1f}",
            True, (0, 0, 0)
        )
        win.blit(text, (10, 10))
        pygame.display.flip()

main()
