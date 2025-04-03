import pygame
import subprocess
import sys

# Интерфейс
WIDTH, HEIGHT = 400, 300
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT_NAME = "arial"

pygame.init()
font = pygame.font.SysFont(FONT_NAME, 22)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Battle Launcher")

def draw_menu():
    screen.fill(WHITE)
    buttons = [
        (" Start Training", "train"),
        (" Manual Play", ["manual_main.py"]),
        (" Watch Best (WIP)", None),
        (" Exit", "exit")
    ]
    for i, (text, command) in enumerate(buttons):
        rect = pygame.Rect(100, 30 + i * 55, 200, 45)
        pygame.draw.rect(screen, BLACK, rect, width=2, border_radius=5)
        label = font.render(text, True, BLACK)
        screen.blit(label, label.get_rect(center=rect.center))
    pygame.display.flip()
    return buttons

def draw_arena_selector():
    screen.fill(WHITE)
    label = font.render("Выберите количество арен:", True, BLACK)
    screen.blit(label, (WIDTH//2 - label.get_width()//2, 30))
    arena_counts = [1, 2, 4, 8]
    buttons = []
    for i, count in enumerate(arena_counts):
        rect = pygame.Rect(50 + i * 80, 100, 60, 50)
        pygame.draw.rect(screen, BLACK, rect, width=2, border_radius=5)
        label = font.render(str(count), True, BLACK)
        screen.blit(label, label.get_rect(center=rect.center))
        buttons.append((rect, count))
    pygame.display.flip()
    return buttons

def main():
    mode = "menu"
    draw_menu()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if mode == "menu":
                    buttons = [
                        (" Start Training", "train"),
                        (" Manual Play", ["manual_main.py"]),
                        (" Watch Best (WIP)", None),
                        (" Exit", "exit")
                    ]
                    for i, (text, command) in enumerate(buttons):
                        rect = pygame.Rect(100, 60 + i * 60, 200, 50)
                        if rect.collidepoint(event.pos):
                            if command == "exit":
                                pygame.quit()
                                sys.exit()
                            elif command == "train":
                                mode = "arena_select"
                                arena_buttons = draw_arena_selector()
                            elif command:
                                pygame.quit()
                                subprocess.run([sys.executable] + command)
                                sys.exit()

                elif mode == "arena_select":
                    for rect, count in draw_arena_selector():
                        if rect.collidepoint(event.pos):
                            pygame.quit()
                            subprocess.run([sys.executable, "train_main.py", str(count)])
                            sys.exit()

if __name__ == "__main__":
    main()
