"""Board encoding, policy indexing, legal-move masking, and plotting helpers."""
import numpy as np
import torch
import chess
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PIECE_TO_PLANE = {
    (chess.WHITE, chess.PAWN): 0, (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2, (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4, (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6, (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8, (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10, (chess.BLACK, chess.KING): 11,
}

def encode_board(board: chess.Board) -> np.ndarray:
    """18x8x8: 12 piece planes, side-to-move, four castling rights, en-passant file plane."""
    x = np.zeros((18, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        x[PIECE_TO_PLANE[(piece.color, piece.piece_type)], rank, file] = 1.0
    x[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    x[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    x[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    x[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    x[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        _, file = divmod(board.ep_square, 8)
        x[17, :, file] = 1.0
    return x

def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

def legal_priors_from_logits(logits: torch.Tensor, board: chess.Board):
    """Return (legal_moves, normalized priors). Promotions share a from-to prior."""
    moves = list(board.legal_moves)
    if not moves:
        return [], []
    indices = torch.tensor([move_to_index(m) for m in moves], device=logits.device, dtype=torch.long)
    legal_logits = logits[indices]
    probs = torch.softmax(legal_logits, dim=0).detach().cpu().numpy()
    return moves, probs.astype(np.float64)

def plot_board(board: chess.Board, title="Chess position", last_move=None, save_path=None):
    """Draw an 8x8 board with Unicode chess pieces using matplotlib."""
    fig, ax = plt.subplots(figsize=(7, 7))
    light, dark = "#f0d9b5", "#b58863"
    symbols = {
        (chess.WHITE, chess.PAWN): "♙", (chess.WHITE, chess.KNIGHT): "♘",
        (chess.WHITE, chess.BISHOP): "♗", (chess.WHITE, chess.ROOK): "♖",
        (chess.WHITE, chess.QUEEN): "♕", (chess.WHITE, chess.KING): "♔",
        (chess.BLACK, chess.PAWN): "♟", (chess.BLACK, chess.KNIGHT): "♞",
        (chess.BLACK, chess.BISHOP): "♝", (chess.BLACK, chess.ROOK): "♜",
        (chess.BLACK, chess.QUEEN): "♛", (chess.BLACK, chess.KING): "♚",
    }
    for rank in range(8):
        for file in range(8):
            color = light if (rank + file) % 2 == 1 else dark
            ax.add_patch(Rectangle((file, rank), 1, 1, color=color))
            piece = board.piece_at(chess.square(file, rank))
            if piece:
                ax.text(file + .5, rank + .5, symbols[(piece.color, piece.piece_type)],
                        ha="center", va="center", fontsize=34,
                        color="#111111" if piece.color == chess.BLACK else "#ffffff",
                        fontfamily="DejaVu Sans")
    if last_move is not None:
        for sq in (last_move.from_square, last_move.to_square):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            ax.add_patch(Rectangle((f, r), 1, 1, fill=False, edgecolor="#2b8a3e", linewidth=3))
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal")
    ax.set_xticks(np.arange(8) + .5, list("abcdefgh"))
    ax.set_yticks(np.arange(8) + .5, [str(i) for i in range(1, 9)])
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=14, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, ax
