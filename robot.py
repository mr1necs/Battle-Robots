import pygame
import math

class Robot:
    def __init__(self, x, y, color, name, agent=None):
        self.spawn_x = x
        self.spawn_y = y
        self.x = x
        self.y = y
        self.angle = 0
        self.color = color
        self.name = name
        self.agent = agent
        self.hp = 3
        self.score = 0
        self.size = 40
        self.walls = {
            "left": {"active": True},
            "right": {"active": True},
            "back": {"active": True}
        }

    def reset(self):
        self.x = self.spawn_x
        self.y = self.spawn_y
        self.hp = 3
        self.score = 0
        self.angle = 0
        self.walls = {
            "left": {"active": True},
            "right": {"active": True},
            "back": {"active": True}
        }

    def draw(self, win):
        # Корпус
        body_surf = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.rect(body_surf, self.color, (0, 0, self.size, self.size))

        # Поворот и отрисовка
        rotated_body = pygame.transform.rotate(body_surf, -self.angle)
        body_rect = rotated_body.get_rect(center=(self.x, self.y))
        win.blit(rotated_body, body_rect)

        # Индикация стенок — сдвинута внутрь
        circle_radius = 5
        offset = self.size * 0.35  # ← СМЕЩЕНИЕ ВНУТРЬ

        cx, cy = self.x, self.y
        angle_rad = math.radians(self.angle)

        wall_offsets = {
            "left": (-math.sin(angle_rad) * offset, math.cos(angle_rad) * offset),
            "right": (math.sin(angle_rad) * offset, -math.cos(angle_rad) * offset),
            "back": (-math.cos(angle_rad) * offset, -math.sin(angle_rad) * offset)
        }

        colors = {
            True: (0, 255, 0),  # зелёный — активна
            False: (255, 0, 0)  # красный — оторвана
        }

        for wall, (dx, dy) in wall_offsets.items():
            wx = int(cx + dx)
            wy = int(cy + dy)
            pygame.draw.circle(win, colors[self.walls[wall]["active"]], (wx, wy), circle_radius)

        # Оружие
        weapon_size = 10
        fx = self.x + math.cos(angle_rad) * (self.size / 2 + 2)
        fy = self.y + math.sin(angle_rad) * (self.size / 2 + 2)

        weapon_surf = pygame.Surface((weapon_size, weapon_size), pygame.SRCALPHA)
        pygame.draw.rect(weapon_surf, (0, 0, 0), (0, 0, weapon_size, weapon_size))
        rotated_weapon = pygame.transform.rotate(weapon_surf, -self.angle)
        weapon_rect = rotated_weapon.get_rect(center=(fx, fy))
        win.blit(rotated_weapon, weapon_rect)

    def move(self, action, opponent, width, height):
        old_x, old_y, old_angle = self.x, self.y, self.angle

        if action == 1:  # вперёд
            self.x += math.cos(math.radians(self.angle)) * 2
            self.y += math.sin(math.radians(self.angle)) * 2
        elif action == 2:  # назад
            self.x -= math.cos(math.radians(self.angle)) * 2
            self.y -= math.sin(math.radians(self.angle)) * 2
        elif action == 3:  # поворот влево
            self.angle -= 5
        elif action == 4:  # поворот вправо
            self.angle += 5

        # Ограничения по полю
        self.x = max(self.size / 2, min(width - self.size / 2, self.x))
        self.y = max(self.size / 2 + 50, min(height - self.size / 2, self.y))

        # Столкновение с другим роботом
        if self.collides_with(opponent):
            self.x, self.y, self.angle = old_x, old_y, old_angle

    def collides_with(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        distance = math.hypot(dx, dy)
        return distance < self.size

    def get_front_tip(self):
        length = self.size / 2 + 5
        fx = self.x + math.cos(math.radians(self.angle)) * length
        fy = self.y + math.sin(math.radians(self.angle)) * length
        return fx, fy

    def get_wall_zones(self):
        half = self.size / 2
        angle_rad = math.radians(self.angle)
        perp_angle = math.radians(self.angle + 90)

        back_x = self.x - math.cos(angle_rad) * half
        back_y = self.y - math.sin(angle_rad) * half

        left_x = self.x + math.cos(perp_angle) * half
        left_y = self.y + math.sin(perp_angle) * half

        right_x = self.x - math.cos(perp_angle) * half
        right_y = self.y - math.sin(perp_angle) * half

        return {
            "left": (left_x, left_y),
            "right": (right_x, right_y),
            "back": (back_x, back_y)
        }
