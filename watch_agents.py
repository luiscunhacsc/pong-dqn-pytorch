import pygame
import torch
from dqn_agent import DQNAgent
from pong import PongGame, WIDTH, HEIGHT, FPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_SIZE = 6
ACTION_SIZE = 3

class WatchAgent(DQNAgent):
    def __init__(self, state_size, action_size, device, model_path):
        super().__init__(state_size, action_size, device)
        self.q_network.load_state_dict(torch.load(model_path, map_location=device))
        self.epsilon = 0.0  # sempre greedy

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

def watch():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong DQN Agents")
    clock = pygame.time.Clock()
    agent_left = WatchAgent(STATE_SIZE, ACTION_SIZE, DEVICE, "dqn_left.pth")
    agent_right = WatchAgent(STATE_SIZE, ACTION_SIZE, DEVICE, "dqn_right.pth")
    game = PongGame()
    running = True
    state = game.reset()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        action_left = agent_left.select_action(state)
        action_right = agent_right.select_action(state)
        next_state, reward, done = game.step(action_left, action_right)
        game.render(screen)
        state = next_state
        if done:
            pygame.time.wait(500)
            state = game.reset()
        clock.tick(FPS)
    pygame.quit()

if __name__ == "__main__":
    watch()
