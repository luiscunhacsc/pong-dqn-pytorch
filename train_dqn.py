import pygame
import numpy as np
import torch
from dqn_agent import DQNAgent
from pong import PongGame, WIDTH, HEIGHT

# Configurações
NUM_EPISODES = 2000
MAX_STEPS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_SIZE = 6
ACTION_SIZE = 3  # 0: parar, 1: cima, 2: baixo

def train():
    agent_left = DQNAgent(STATE_SIZE, ACTION_SIZE, DEVICE)
    agent_right = DQNAgent(STATE_SIZE, ACTION_SIZE, DEVICE)
    game = PongGame()
    scores = []
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Treino Pong DQN")
    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()

    for episode in range(1, NUM_EPISODES + 1):
        # Atualizar indicação do episódio de 10 em 10
        if episode % 10 == 0:
            print(f"Episódio {episode} em curso...")
        state = game.reset()
        done = False
        total_reward = [0, 0]
        step = 0
        # Mostrar agentes a jogar de 100 em 100 episódios
        if episode % 100 == 0:
            show_state = game.reset()
            show_done = False
            # Obter pontuação inicial
            initial_score = sum(game.score)
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                # Usar greedy (epsilon=0) para ver comportamento aprendido
                action_left = agent_left.q_network(torch.FloatTensor(show_state).unsqueeze(0).to(DEVICE)).argmax().item()
                action_right = agent_right.q_network(torch.FloatTensor(show_state).unsqueeze(0).to(DEVICE)).argmax().item()
                next_state, reward, show_done = game.step(action_left, action_right)
                game.render(screen)
                # Mostrar episódio no topo
                ep_text = font.render(f"Episódio {episode}", True, (255,255,0))
                screen.blit(ep_text, (WIDTH//2 - ep_text.get_width()//2, 50))
                pygame.display.flip()
                show_state = next_state
                clock.tick(60)
                # Critério de paragem: soma dos pontos aumentou em pelo menos 5
                if sum(game.score) - initial_score >= 5:
                    break
                if show_done:
                    pygame.time.wait(400)
                    show_state = game.reset()
                    show_done = False
        # Treino normal
        while not done and step < MAX_STEPS:
            action_left = agent_left.select_action(state)
            action_right = agent_right.select_action(state)
            next_state, reward, done = game.step(action_left, action_right)
            agent_left.remember(state, action_left, reward[0], next_state, done)
            agent_right.remember(state, action_right, reward[1], next_state, done)
            agent_left.learn()
            agent_right.learn()
            state = next_state
            total_reward[0] += reward[0]
            total_reward[1] += reward[1]
            # Atualizar o número do episódio no ecrã de 10 em 10 episódios (mas não mostrar o jogo)
            if episode % 10 == 0:
                screen.fill((0,0,0))
                ep_text = font.render(f"Episódio {episode}", True, (255,255,0))
                screen.blit(ep_text, (WIDTH//2 - ep_text.get_width()//2, HEIGHT//2 - ep_text.get_height()//2))
                pygame.display.flip()
            step += 1
        scores.append((total_reward[0], total_reward[1]))
        if episode % 100 == 0:
            avg_left = np.mean([s[0] for s in scores[-100:]])
            avg_right = np.mean([s[1] for s in scores[-100:]])
            print(f"Episódio {episode} | Média últimos 100 episódios: Esquerda={avg_left:.2f}, Direita={avg_right:.2f}")
    # Salvar modelos treinados
    torch.save(agent_left.q_network.state_dict(), "dqn_left.pth")
    torch.save(agent_right.q_network.state_dict(), "dqn_right.pth")
    print("Treino terminado e modelos salvos!")
    pygame.quit()

if __name__ == "__main__":
    train()
