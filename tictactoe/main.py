import torch

from ttt_rl.game import GameStatus
from ttt_rl.intelligence import NeuralNet, play_game


def train():
    n_games = 128

    model = NeuralNet()
    optimizer = torch.optim.Adam(model.parameters())

    for _ in range(n_games):
        transitions, game_result = play_game(model, model)
        reward = (
            1 if game_result == GameStatus.X_WIN
            else 0 if game_result == GameStatus.DRAW
            else -1
        )
        loss = torch.stack([
            -reward * torch.log(transition.action_prob)
            for transition in transitions
        ], dim=0).sum(dim=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.detach(), game_result)


if __name__ == "__main__":
    train()
