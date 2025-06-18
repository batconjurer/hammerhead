use std::fmt::{Display, Formatter};

use board::Board;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

use crate::game::space::{Role, Square};

pub mod board;
pub mod space;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct PreviousBoards(pub FxHashSet<Board>);

#[derive(Copy, Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum Status {
    AttackersWin,
    #[default]
    Ongoing,
    DefendersWin,
}

impl Display for Status {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::AttackersWin => f.write_str("Attackers win"),
            Status::Ongoing => f.write_str("Game ongoing"),
            Status::DefendersWin => f.write_str("Defenders win"),
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
    pub fn valid(&self) -> anyhow::Result<()> {
        if std::cmp::max(self.from.x, self.from.y) > 10 {
            return Err(anyhow::Error::msg(
                "The piece to be moved must be on a square on the board.",
            ));
        }
        if std::cmp::max(self.to.x, self.to.y) > 10 {
            return Err(anyhow::Error::msg(
                "The piece must be moved to a square on the board.",
            ));
        }
        let x_diff = self.from.x as i32 - self.to.x as i32;
        let y_diff = self.from.y as i32 - self.to.y as i32;

        if x_diff != 0 && y_diff != 0 {
            return Err(anyhow::Error::msg(
                "play: you can only play in a straight line",
            ));
        }

        if x_diff == 0 && y_diff == 0 {
            return Err(anyhow::Error::msg("play: you have to change location"));
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Game {
    pub status: Status,
    pub previous_boards: PreviousBoards,
    pub history: Vec<Board>,
    pub ahead: Vec<Board>,
    pub turn: Role,
    pub current_board: Board,
}

impl Display for Game {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("Status: {}\n", self.status))?;
        f.write_str(&format!("Turn: {}\n", self.turn))?;
        self.current_board.fmt(f)
    }
}
impl Game {
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
