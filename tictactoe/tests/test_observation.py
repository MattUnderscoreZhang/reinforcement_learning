from ttt_rl.game import init_game_state
from ttt_rl.intelligence import get_observation


def test_get_observation():
    game_state = init_game_state()
    observation = get_observation(game_state)
