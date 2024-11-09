from dataclasses import dataclass
from enum import Enum


class SpaceState(Enum):
    EMPTY = 1
    X_MOVE = 2
    O_MOVE = 3


class Player(Enum):
    X_PLAYER = 1
    O_PLAYER = 2


class GameStatus(Enum):
    IN_PROGRESS = 1
    X_WIN = 2
    O_WIN = 3
    DRAW = 4


@dataclass
class GameState:
    board: list[list[SpaceState]]
    current_player: Player
    game_status: GameStatus


def init_game_state() -> GameState:
    return GameState(
        board=[
            [SpaceState.EMPTY, SpaceState.EMPTY, SpaceState.EMPTY],
            [SpaceState.EMPTY, SpaceState.EMPTY, SpaceState.EMPTY],
            [SpaceState.EMPTY, SpaceState.EMPTY, SpaceState.EMPTY],
        ],
        current_player=Player.X_PLAYER,
        game_status=GameStatus.IN_PROGRESS,
    )


def check_if_player_won(board: list[list[SpaceState]], player_move: SpaceState) -> bool:
    return any(
        all(board[row][col] == player_move for col in range(3))
        for row in range(3)
    ) or any(
        all(board[row][col] == player_move for row in range(3))
        for col in range(3)
    ) or all(
        board[i][i] == player_move for i in range(3)
    ) or all(
        board[i][2 - i] == player_move for i in range(3)
    )

def check_game_status(board: list[list[SpaceState]]) -> GameStatus:
    if check_if_player_won(board, SpaceState.X_MOVE):
        return GameStatus.X_WIN
    if check_if_player_won(board, SpaceState.O_MOVE):
        return GameStatus.O_WIN
    if all(board[row][col] != SpaceState.EMPTY for col in range(3) for row in range(3)):
        return GameStatus.DRAW
    return GameStatus.IN_PROGRESS


def make_move(game_state: GameState, action: int) -> GameState:
    row = action // 3
    col = action % 3
    new_board = game_state.board
    if new_board[row][col] != SpaceState.EMPTY:
        raise ValueError("Invalid move")
    new_board[row][col] = (
        SpaceState.X_MOVE if game_state.current_player == Player.X_PLAYER
        else SpaceState.O_MOVE
    )
    return GameState(
        board=new_board,
        current_player=(
            Player.O_PLAYER
            if game_state.current_player == Player.X_PLAYER
            else Player.X_PLAYER
        ),
        game_status=check_game_status(new_board),
    )
