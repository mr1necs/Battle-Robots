import pygame
import torch
import math
import csv
import os
import sys
import shutil
import matplotlib.pyplot as plt
import pandas as pd

from robot import Robot
from agent import Agent
from network import Net
from utils import get_state, check_hit, draw_ui
train_count = 0

# === Настройки симуляции ===
WIDTH, HEIGHT = 400, 450
ARENA_Y_OFFSET = 50
FPS = 60
BATTLE_DURATION = FPS * 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROBOT_SIZE = 40

# === Режим запуска и директория ===
fresh_start = "--fresh" in sys.argv
base_run_dir = "runs"
os.makedirs(base_run_dir, exist_ok=True)

# Создаём папку runs/N
if fresh_start:
    run_id = 1
    while os.path.exists(os.path.join(base_run_dir, f"{run_id}")):
        run_id += 1
    run_dir = os.path.join(base_run_dir, str(run_id))
    os.makedirs(run_dir)
    print(f" Новый запуск: {run_dir}")
else:
    existing = sorted([int(f) for f in os.listdir(base_run_dir) if f.isdigit()])
    run_dir = os.path.join(base_run_dir, str(existing[-1]) if existing else "1")
    print(f" Продолжаем обучение: {run_dir}")

# Путь к логам
log_path = os.path.join(run_dir, "log.csv")

# === Начальная инициализация ===
best_score_sum = -float("inf")
round_count = 0

# Восстанавливаем номер раунда из лога
if os.path.exists(log_path):
    with open(log_path) as f:
        round_count = sum(1 for _ in f) - 1

# Создаём лог при fresh-запуске
if fresh_start or not os.path.exists(log_path):
    os.makedirs(run_dir, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "r1_score", "r2_score", "r1_hp", "r2_hp"])

# Проверка, в углу ли робот
def in_corner(robot):
    margin = 30
    x, y = robot.x, robot.y
    return (x < margin and y < margin) or \
           (x > WIDTH - margin - ROBOT_SIZE and y < margin) or \
           (x < margin and y > HEIGHT - margin - ROBOT_SIZE) or \
           (x > WIDTH - margin - ROBOT_SIZE and y > HEIGHT - margin - ROBOT_SIZE)

def main():
    global best_score_sum, round_count

    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    net = Net().to(DEVICE)
    agent = Agent(net)

    # Загружаем последнюю модель, если есть
    model_files = [f for f in os.listdir(run_dir) if f.startswith("best_model") and f.endswith(".pt")]
    if model_files:
        latest = sorted(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
        net.load_state_dict(torch.load(os.path.join(run_dir, latest)))
        print(f" Загружена модель: {latest}")

    MARGIN = 40
    arena_top = ARENA_Y_OFFSET
    arena_height = HEIGHT - ARENA_Y_OFFSET

    robot1 = Robot(MARGIN, arena_top + MARGIN, (0, 120, 255), "R1", agent)
    robot2 = Robot(WIDTH - MARGIN, arena_top + arena_height - MARGIN, (255, 60, 60), "R2", agent)

    timer = 0
    run = True

    while run:
        clock.tick(FPS)
        timer += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Состояния
        state1 = get_state(robot1, robot2)
        state2 = get_state(robot2, robot1)

        # Действия
        action1 = agent.select_action(state1)
        action2 = agent.select_action(state2)

        robot1.move(action1, robot2, WIDTH, HEIGHT)
        robot2.move(action2, robot1, WIDTH, HEIGHT)

        # Поощрения и штрафы
        if action1 in [1, 2]: robot1.score += 0.1
        if action2 in [1, 2]: robot2.score += 0.1
        if action1 in [3, 4]: robot1.score -= 0.2
        if action2 in [3, 4]: robot2.score -= 0.2
        if robot1.collides_with(robot2):
            robot1.score -= 0.5
            robot2.score -= 0.5
        if in_corner(robot1): robot1.score -= 0.5
        if in_corner(robot2): robot2.score -= 0.5

        # Попадания
        r1_hit, r2_penalty = check_hit(robot1, robot2)
        r2_hit, r1_penalty = check_hit(robot2, robot1)
        robot1.score += r1_hit
        robot2.score += r2_hit

        # Replay Buffer
        next_state1 = get_state(robot1, robot2)
        next_state2 = get_state(robot2, robot1)
        agent.store(state1, action1, r1_hit + r1_penalty, next_state1)
        agent.store(state2, action2, r2_hit + r2_penalty, next_state2)

        # Отрисовка
        win.fill((240, 240, 240))
        draw_ui(win, timer, robot1, robot2)
        pygame.draw.rect(win, (150, 150, 150), (0, ARENA_Y_OFFSET, WIDTH, WIDTH), 5)
        robot1.draw(win)
        robot2.draw(win)
        pygame.display.flip()

        # Конец боя
        if timer >= BATTLE_DURATION or robot1.hp <= 0 or robot2.hp <= 0:
            round_count += 1

            # Лог
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([round_count, robot1.score, robot2.score, robot1.hp, robot2.hp])

            # Сохраняем модель, если лучший бой
            total_score = robot1.score + robot2.score
            if total_score > best_score_sum:
                best_score_sum = total_score
                torch.save(net.state_dict(), os.path.join(run_dir, f"best_model_round_{round_count}.pt"))
                print(f" Лучший бой! Round {round_count}, Score: {total_score:.1f}")

            # График каждые 100
            if round_count % 100 == 0:
                df = pd.read_csv(log_path)
                plt.figure(figsize=(10, 4))
                plt.plot(df["round"], df["r1_score"], label="R1 Score", color="blue")
                plt.plot(df["round"], df["r2_score"], label="R2 Score", color="red")
                plt.plot(df["round"], df["r1_hp"], "--", label="R1 HP", color="blue", alpha=0.5)
                plt.plot(df["round"], df["r2_hp"], "--", label="R2 HP", color="red", alpha=0.5)
                plt.xlabel("Раунд")
                plt.ylabel("Очки / HP")
                plt.title(f"Обучение до {round_count} боёв")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, f"plot_{round_count}.png"))
                plt.close()
                print(f" Сохранён график: plot_{round_count}.png")

            agent.train()
            robot1.reset()
            robot2.reset()
            timer = 0

    pygame.quit()

if __name__ == "__main__":
    main()
