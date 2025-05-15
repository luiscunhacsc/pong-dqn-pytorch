import pygame
import random

# Constantes do jogo
WIDTH, HEIGHT = 640, 480
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 80
BALL_SIZE = 15
PADDLE_SPEED = 5
BALL_SPEED = 5
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Paddle:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.speed = 0

    def move(self, dy):
        self.rect.y += dy
        self.rect.y = max(0, min(self.rect.y, HEIGHT - PADDLE_HEIGHT))

    def update(self):
        self.move(self.speed)

class Ball:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.reset()

    def reset(self):
        self.rect.x = WIDTH // 2 - BALL_SIZE // 2
        self.rect.y = HEIGHT // 2 - BALL_SIZE // 2
        angle = random.choice([random.uniform(-0.25, 0.25), random.uniform(0.75, 1.25)]) * 3.14
        self.vx = BALL_SPEED * random.choice([-1, 1])
        self.vy = BALL_SPEED * random.choice([-1, 1])

    def update(self):
        self.rect.x += self.vx
        self.rect.y += self.vy
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.vy = -self.vy

class PongGame:
    def __init__(self):
        self.left_paddle = Paddle(10, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        self.right_paddle = Paddle(WIDTH - 20, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        self.ball = Ball()
        self.score = [0, 0]

    def reset(self):
        self.left_paddle.rect.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.right_paddle.rect.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.ball.reset()
        return self.get_state()

    def get_state(self):
        # Estado: posição e velocidade da bola, posição das raquetes
        return [
            self.left_paddle.rect.y / HEIGHT,
            self.right_paddle.rect.y / HEIGHT,
            self.ball.rect.x / WIDTH,
            self.ball.rect.y / HEIGHT,
            self.ball.vx / BALL_SPEED,
            self.ball.vy / BALL_SPEED
        ]

    def step(self, action_left, action_right):
        # Ações: 0 = parar, 1 = cima, 2 = baixo
        self.left_paddle.speed = 0
        self.right_paddle.speed = 0
        if action_left == 1:
            self.left_paddle.speed = -PADDLE_SPEED
        elif action_left == 2:
            self.left_paddle.speed = PADDLE_SPEED
        if action_right == 1:
            self.right_paddle.speed = -PADDLE_SPEED
        elif action_right == 2:
            self.right_paddle.speed = PADDLE_SPEED

        self.left_paddle.update()
        self.right_paddle.update()
        self.ball.update()

        reward = [0, 0]
        done = False

        # Colisão com raquetes
        if self.ball.rect.colliderect(self.left_paddle.rect):
            self.ball.vx = abs(self.ball.vx)
        if self.ball.rect.colliderect(self.right_paddle.rect):
            self.ball.vx = -abs(self.ball.vx)

        # Golo
        if self.ball.rect.left <= 0:
            self.score[1] += 1
            reward = [-1, 1]
            done = True
            self.ball.reset()
        elif self.ball.rect.right >= WIDTH:
            self.score[0] += 1
            reward = [1, -1]
            done = True
            self.ball.reset()

        return self.get_state(), reward, done

    def render(self, screen):
        screen.fill(BLACK)
        pygame.draw.rect(screen, WHITE, self.left_paddle.rect)
        pygame.draw.rect(screen, WHITE, self.right_paddle.rect)
        pygame.draw.ellipse(screen, WHITE, self.ball.rect)
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"{self.score[0]} : {self.score[1]}", True, WHITE)
        screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))
        pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong RL")
    clock = pygame.time.Clock()
    game = PongGame()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        # Controlos manuais para testar o ambiente
        action_left = 0
        if keys[pygame.K_w]:
            action_left = 1
        elif keys[pygame.K_s]:
            action_left = 2
        action_right = 0
        if keys[pygame.K_UP]:
            action_right = 1
        elif keys[pygame.K_DOWN]:
            action_right = 2
        _, _, _ = game.step(action_left, action_right)
        game.render(screen)
        clock.tick(FPS)
    pygame.quit()

if __name__ == "__main__":
    main()
