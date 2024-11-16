from dataclasses import dataclass
import torch

from ttt_rl.game import GameState, GameStatus, SpaceState, init_game_state, make_move


def get_observation(game_state: GameState) -> torch.Tensor:
    return torch.tensor([
        space_state.value
        for game_row in game_state.board
        for space_state in game_row
    ], dtype=torch.float32)


def get_mask(game_state: GameState) -> torch.Tensor:
    return torch.tensor([
        space_state == SpaceState.EMPTY
        for game_row in game_state.board
        for space_state in game_row
    ], dtype=torch.float32)


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
            torch.nn.Softmax(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        action_probs = self.model(observation)
        return action_probs


@dataclass
class Transition:
    observation: torch.Tensor
    action_prob: torch.Tensor
    next_observation: torch.Tensor


def play_game(x_model: NeuralNet, o_model: NeuralNet) -> tuple[list[Transition], GameStatus]:
    game_state = init_game_state()
    current_model = x_model
    transitions = []
    while game_state.game_status == GameStatus.IN_PROGRESS:
        observation = get_observation(game_state)
        mask = get_mask(game_state)
        action_probs = current_model(observation) * mask
        action = int(torch.argmax(action_probs))
        game_state = make_move(game_state, action)
        transitions.append(Transition(observation, action_probs[action], get_observation(game_state)))
        current_model = o_model if current_model == x_model else x_model
    return transitions, game_state.game_status
