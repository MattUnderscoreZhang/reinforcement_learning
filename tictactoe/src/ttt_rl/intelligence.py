from dataclasses import dataclass
import torch

from ttt_rl.game import GameState, GameStatus, init_game_state, make_move


def get_observation(game_state: GameState):
    return torch.tensor([
        space_state.value
        for game_row in game_state.board
        for space_state in game_row
    ])


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_hidden_layers = 128
        self.model = torch.nn.Sequential(
            torch.nn.Linear(9, n_hidden_layers),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_layers, n_hidden_layers),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_layers, n_hidden_layers),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_layers, 9),
            torch.nn.Softmax(dim=0),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        action_logits = self.model(observation)
        return action_logits


@dataclass
class Transition:
    observation: torch.Tensor
    action_logit: torch.Tensor
    action: int
    next_observation: torch.Tensor


def play_game(x_model: NeuralNet, o_model: NeuralNet) -> GameStatus:
    game_state = init_game_state()
    current_model = x_model
    transitions = []
    while game_state.game_status == GameStatus.IN_PROGRESS:
        observation = get_observation(game_state)
        action_logits = current_model(observation)
        action = int(torch.argmax(action_logits))
        game_state = make_move(game_state, action)
        transitions.append((observation, action_logits, get_observation(game_state)))
        current_model = o_model if current_model == x_model else x_model
    return game_state.game_status


def train():
    n_games_in_batch = 512
    n_batches = 100_000

    model = NeuralNet()
    x_model = model
    o_model = model
    x_optimizer = torch.optim.adam.Adam(x_model.parameters())
    o_optimizer = torch.optim.adam.Adam(o_model.parameters())

    for _ in range(n_batches):
        # TODO: algorithm doesn't work for discrete reward
        game_results = torch.Tensor([
            play_game(x_model, o_model)
            for _ in range(n_games_in_batch)
        ])

        x_reward = torch.sum(torch[
            2 if result == GameStatus.X_WIN
            else 1 if result == GameStatus.DRAW
            else 0
            for result in game_results
        ])
        x_loss = -x_reward
        x_optimizer.zero_grad()
        x_loss.backward()
        x_optimizer.step()

        o_reward = -x_reward
        o_optimizer.zero_grad()
        x_loss = -o_reward
        o_loss.backward()
        o_optimizer.step()
