import multiprocessing as mp
import torch
import sys
import os
import pygame
import csv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from network import Net
from agent import Agent
from robot import Robot
from utils import get_state, check_hit

WIDTH, HEIGHT = 400, 450
FPS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def arena_worker(agent, queue, index, enable_logging=False, log_dir=None):
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Arena {index}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18)

    MARGIN = 60

    robot1 = Robot(MARGIN, MARGIN, (0, 120, 255), "R1")
    robot2 = Robot(WIDTH - MARGIN, HEIGHT - MARGIN, (255, 60, 60), "R2", agent)
    timer = 0
    best_score = -float("inf")
    last_saved_plot = 0
    BATTLE_DURATION = FPS * 30

    # === Инициализация round_num из лога
    round_num = 1
    if enable_logging and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "log.csv")

        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "r1_score", "r2_score", "r1_hp", "r2_hp", "timestamp", "total_score"])

        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                last_round = int(lines[-1].split(",")[0]) if len(lines) > 1 else 0
        except:
            last_round = 0

        round_num = last_round + 1

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(FPS)
        timer += 1

        # Получаем действия и состояния
        state1 = get_state(robot1, robot2)
        state2 = get_state(robot2, robot1)
        action1 = agent.select_action(state1)
        action2 = agent.select_action(state2)
        robot1.move(action1, robot2, WIDTH, HEIGHT)
        robot2.move(action2, robot1, WIDTH, HEIGHT)

        # Бонусы и штрафы
        if action1 in [1, 2]: robot1.score += 0.1
        if action2 in [1, 2]: robot2.score += 0.1
        if action1 in [3, 4]: robot1.score -= 0.2
        if action2 in [3, 4]: robot2.score -= 0.2

        if robot1.collides_with(robot2):
            robot1.score -= 0.5
            robot2.score -= 0.5

        def in_corner(robot):
            margin = 30
            return (
                (robot.x < margin and robot.y < margin) or
                (robot.x > WIDTH - margin and robot.y < margin) or
                (robot.x < margin and robot.y > HEIGHT - margin) or
                (robot.x > WIDTH - margin and robot.y > HEIGHT - margin)
            )

        if in_corner(robot1): robot1.score -= 0.5
        if in_corner(robot2): robot2.score -= 0.5

        # Проверка попаданий
        r1_reward, r2_penalty = check_hit(robot1, robot2)
        r2_reward, r1_penalty = check_hit(robot2, robot1)

        next_state1 = get_state(robot1, robot2)
        next_state2 = get_state(robot2, robot1)
        queue.put((state1, action1, r1_reward + r1_penalty, next_state1))
        queue.put((state2, action2, r2_reward + r2_penalty, next_state2))

        # Отрисовка
        win.fill((240, 240, 240))
        pygame.draw.rect(win, (150, 150, 150), (0, 50, WIDTH, WIDTH), 5)
        robot1.draw(win)
        robot2.draw(win)

        ui_text = font.render(
            f"Time: {timer // FPS}s | R1 HP: {robot1.hp} Score: {robot1.score:.1f} | R2 HP: {robot2.hp} Score: {robot2.score:.1f}",
            True, (0, 0, 0)
        )
        win.blit(ui_text, (10, 10))
        pygame.display.flip()

        # Конец боя
        if timer >= BATTLE_DURATION or robot1.hp <= 0 or robot2.hp <= 0:
            total_score = robot1.score + robot2.score

            if enable_logging and log_dir:
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        round_num,
                        robot1.score, robot2.score,
                        robot1.hp, robot2.hp,
                        datetime.now().isoformat(),
                        total_score
                    ])

                # Сохраняем лучшую модель
                if total_score > best_score:
                    best_score = total_score
                    torch.save(agent.net.state_dict(), os.path.join(log_dir, f"best_model_round_{round_num}.pt"))
                    print(f"Лучший бой! Раунд: {round_num} | Суммарный score: {total_score:.1f}")

                # График каждые 100 боёв
                if round_num % 100 == 0:
                    df = pd.read_csv(log_path)
                    plt.figure(figsize=(10, 4))
                    plt.plot(df["round"], df["r1_score"], label="R1 Score", color="blue")
                    plt.plot(df["round"], df["r2_score"], label="R2 Score", color="red")
                    plt.plot(df["round"], df["r1_hp"], linestyle="--", label="R1 HP", color="blue", alpha=0.5)
                    plt.plot(df["round"], df["r2_hp"], linestyle="--", label="R2 HP", color="red", alpha=0.5)
                    best_row = df.loc[df["total_score"].idxmax()]
                    plt.scatter(best_row["round"], best_row["total_score"], color="black", label="Best Score", zorder=5)
                    plt.title(f"Обучение до {round_num} раундов")
                    plt.xlabel("Раунд")
                    plt.ylabel("Счёт / HP")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(log_dir, f"plot_{round_num}.png"))
                    plt.close()

            robot1.reset()
            robot2.reset()
            timer = 0
            round_num += 1



    pygame.quit()


def main():
    if len(sys.argv) < 2:
        print("Укажите количество арен: python train_main.py 1 [модель.pt]")
        return

    num_arenas = int(sys.argv[1])
    manual_model = sys.argv[2] if len(sys.argv) >= 3 else None
    log_dir = None

    print(f"Запуск {num_arenas} арен(ы)")
    net = Net().to(DEVICE)
    agent = Agent(net)

    if manual_model:
        try:
            net.load_state_dict(torch.load(manual_model))
            print(f"Загружена модель: {manual_model}")
            base_dir = os.path.dirname(manual_model)
            i = 1
            while os.path.exists(f"{base_dir}.{i}"):
                i += 1
            log_dir = f"{base_dir}.{i}"
            os.makedirs(log_dir)
            print(f"Создана новая папка логов: {log_dir}")
            print(f"Продолжение обучения в папке: {log_dir}")
        except Exception as e:
            print(f"Не удалось загрузить модель '{manual_model}': {e}")

    if not log_dir and num_arenas == 1:
        os.makedirs("runs", exist_ok=True)
        run_id = 1
        while os.path.exists(os.path.join("runs", str(run_id))):
            run_id += 1
        log_dir = os.path.join("runs", str(run_id))
        print(f"Создана новая папка логов: {log_dir}")

    p = mp.Process(target=arena_worker, args=(agent, mp.Queue(), 0, True if num_arenas == 1 else False, log_dir))
    p.start()
    p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
