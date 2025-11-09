use std::fmt::{Display, Formatter};

use board::Board;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::alpha_beta::alphabeta;
use crate::alpha_beta::heuristic::HeuristicPolicy;
use crate::game::space::{Role, Square};
pub use crate::game::symmetries::{NormalizedBoardMap, NormalizedBoards};
use crate::game_tree::{GameSummary, GameTreeNode};
use crate::mcts::scaled_i64_to_float;

pub mod board;
pub mod heuristics;
pub mod space;
mod symmetries;

#[derive(Error, Debug)]
pub enum PlayError {
    #[error("The game has to be ongoing to play")]
    GameFinished,
    #[error("The piece was either moved from or to a square not on the board")]
    InvalidSquare,
    #[error("Pieces may only be moved in straight lines")]
    StraightLine,
    #[error("The start and end squares for a move piece cannot be the same")]
    DidntMove,
    #[error("Attempted to move a piece belonging to the opposite player")]
    WrongTurn,
    #[error("Attempted to move a piece through another piece")]
    MoveThroughPiece,
    #[error("Only the king may move to a restricted square")]
    RestrictedSquare,
    #[error("A defender can't repeat a board position")]
    RepeatedPosition,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct PreviousBoards(pub FxHashSet<Board>);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PositionsTracker {
    Previous(PreviousBoards),
    Counter(usize),
}

impl PositionsTracker {
    pub fn len(&self) -> usize {
        match self {
            PositionsTracker::Previous(prev) => prev.0.len(),
            PositionsTracker::Counter(length) => *length,
        }
    }

    pub fn insert(&mut self, board: &Board) {
        match self {
            PositionsTracker::Previous(prev) => {
                _ = prev.0.insert(board.clone());
            }
            PositionsTracker::Counter(moves) => *moves += 1,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize, Hash)]
pub enum Status {
    AttackersWin,
    #[default]
    Ongoing,
    DefendersWin,
    Draw,
}

impl Display for Status {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::AttackersWin => f.write_str("Attackers win"),
            Status::Ongoing => f.write_str("Game ongoing"),
            Status::DefendersWin => f.write_str("Defenders win"),
            Status::Draw => f.write_str("Draw"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Play {
    pub role: Role,
    pub from: Square,
    pub to: Square,
}

impl Play {
    pub fn valid(&self) -> Result<(), PlayError> {
        if std::cmp::max(self.from.x, self.from.y) > 10 {
            return Err(PlayError::InvalidSquare);
        }
        if std::cmp::max(self.to.x, self.to.y) > 10 {
            return Err(PlayError::InvalidSquare);
        }
        let x_diff = self.from.x as i32 - self.to.x as i32;
        let y_diff = self.from.y as i32 - self.to.y as i32;

        if x_diff != 0 && y_diff != 0 {
            return Err(PlayError::StraightLine);
        }

        if x_diff == 0 && y_diff == 0 {
            return Err(PlayError::DidntMove);
        }

        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct EngineRole {
    engine: HeuristicPolicy,
    role: Role,
}

impl From<Role> for EngineRole {
    fn from(role: Role) -> Self {
        Self {
            engine: Default::default(),
            role,
        }
    }
}

/// A UI friendly version of a game for playing on the CLI
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveGame {
    pub status: Status,
    pub previous_boards: PositionsTracker,
    pub history: Vec<Board>,
    pub ahead: Vec<Board>,
    pub turn: Role,
    pub current_board: Board,
    pub engine: Option<EngineRole>,
}

impl Default for LiveGame {
    fn default() -> Self {
        Self {
            status: Default::default(),
            previous_boards: PositionsTracker::Previous(Default::default()),
            history: vec![],
            ahead: vec![],
            turn: Default::default(),
            current_board: Default::default(),
            engine: None,
        }
    }
}

impl Display for LiveGame {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("Status: {}\n", self.status))?;
        f.write_str(&format!("Turn: {}\n", self.turn))?;
        self.current_board.fmt(f)
    }
}

impl From<&mut LiveGame> for GameTreeNode {
    fn from(game: &mut LiveGame) -> Self {
        GameTreeNode {
            status: game.status,
            previous_boards: PositionsTracker::Counter(game.previous_boards.len()),
            turn: game.turn,
            current_board: game.current_board.clone(),
        }
    }
}

impl LiveGame {
    /// Play a move and update the game state
    pub fn play(&mut self, play: &Play) -> anyhow::Result<()> {
        let current = self.current_board.clone();
        let (_, status) = self
            .current_board
            .play(play, &self.status, &mut self.previous_boards)?;
        self.history.push(current);
        self.ahead.clear();
        self.turn = self.turn.opposite();
        self.status = status;
        Ok(())
    }

    /// If the game has an engine attached, use it to
    /// make a move if it is the engine's turn. Returns
    /// a boolean indicating if the engine played or not.
    pub fn engine_play(&mut self) -> bool {
        let Some(EngineRole { engine, role }) = self.engine else {
            return false;
        };
        if self.turn != role {
            return false;
        }
        if self.status != Status::Ongoing {
            return false;
        }

        let root = GameTreeNode::from(&mut *self);
        let (score, next) = match root.turn {
            Role::Attacker => root
                .get_children()
                .into_iter()
                .map(|c| (alphabeta::<GameSummary, _, _>(&c, &engine, 3), c))
                .max_by_key(|c| c.0)
                .unwrap(),
            Role::Defender => root
                .get_children()
                .into_iter()
                .map(|c| (alphabeta::<GameSummary, _, _>(&c, &engine, 3), c))
                .max_by_key(|c| c.0)
                .unwrap(),
        };
        println!(
            "Evaluation of best position: {}",
            scaled_i64_to_float(score)
        );
        println!("Done");
        let current = self.current_board.clone();
        self.history.push(current.clone());
        self.previous_boards.insert(&current);
        self.ahead.clear();
        self.turn = next.turn;
        self.status = next.status;
        self.current_board = next.current_board;
        true
    }

    /// Undo a move
    pub fn undo(&mut self) {
        if let Some(mut board) = self.history.pop() {
            std::mem::swap(&mut self.current_board, &mut board);
            self.ahead.push(board);
            self.turn = self.turn.opposite();
        }
    }

    /// Redo a move
    pub fn redo(&mut self) {
        if let Some(mut board) = self.ahead.pop() {
            std::mem::swap(&mut self.current_board, &mut board);
            self.history.push(board);
            self.turn = self.turn.opposite();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that a play from or to a square not in the board
    /// bounds results in an error
    #[test]
    fn test_play_invalid_squares() {
        assert!(
            Play {
                role: Role::Defender,
                from: Square { x: 0, y: 11 },
                to: Square { x: 0, y: 0 }
            }
            .valid()
            .is_err()
        );
        assert!(
            Play {
                role: Role::Defender,
                from: Square { x: 11, y: 0 },
                to: Square { x: 0, y: 0 }
            }
            .valid()
            .is_err()
        );
        assert!(
            Play {
                role: Role::Defender,
                from: Square { x: 0, y: 0 },
                to: Square { x: 0, y: 11 }
            }
            .valid()
            .is_err()
        );
        assert!(
            Play {
                role: Role::Defender,
                from: Square { x: 0, y: 0 },
                to: Square { x: 11, y: 0 }
            }
            .valid()
            .is_err()
        );
    }

    /// Check that we enforce moving only in a straight line
    /// and cannot remain in place
    #[test]
    fn test_straight_line() {
        assert!(
            Play {
                role: Default::default(),
                from: Square { x: 0, y: 0 },
                to: Square { x: 10, y: 10 },
            }
            .valid()
            .is_err()
        );
        assert!(
            Play {
                role: Default::default(),
                from: Square { x: 5, y: 5 },
                to: Square { x: 5, y: 5 },
            }
            .valid()
            .is_err()
        );
        assert!(
            Play {
                role: Default::default(),
                from: Square { x: 6, y: 6 },
                to: Square { x: 5, y: 5 },
            }
            .valid()
            .is_err()
        );
        assert!(
            Play {
                role: Default::default(),
                from: Square { x: 6, y: 6 },
                to: Square { x: 9, y: 6 },
            }
            .valid()
            .is_ok()
        );
        assert!(
            Play {
                role: Default::default(),
                from: Square { x: 6, y: 6 },
                to: Square { x: 6, y: 0 },
            }
            .valid()
            .is_ok()
        )
    }
}
