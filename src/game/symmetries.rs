//! Hnefatafl is symmetric with respect to the symmetries of the square,
//! the groupd D8. This contains utilities to exploit that symmetry.

use crate::game::board::Board;
use crate::game::space::{Space, Square};

/// Two elements that generate D8
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum D8Generator {
    F,
    FR,
}

impl D8Generator {
    /// Apply a generator of D8 to the board
    pub fn apply(&self, board: &mut Board) {
        match self {
            D8Generator::F => {
                let mut new_board = Board::empty();
                for square in Square::iter() {
                    let space = board.get(&square);
                    if matches!(space, Space::Occupied(_) | Space::King) {
                        new_board.set(
                            &Square {
                                x: square.x,
                                y: 10 - square.y,
                            },
                            space,
                        );
                    }
                }
                *board = new_board;
            }
            D8Generator::FR => {
                let mut new_board = Board::empty();
                for square in Square::iter() {
                    let space = board.get(&square);
                    if matches!(space, Space::Occupied(_) | Space::King) {
                        new_board.set(
                            &Square {
                                x: square.y,
                                y: square.x,
                            },
                            space,
                        );
                    }
                }
                *board = new_board;
            }
        }
    }
}

/// An element of D8 expressed as a word in the two chosen generators
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct D8Element([Option<D8Generator>; 4]);

impl D8Element {
    /// Apply a D8 element to the board
    pub fn apply(&self, board: &mut Board) {
        for generator in self.0 {
            if let Some(g) = generator {
                g.apply(board)
            } else {
                return;
            }
        }
    }
}

/// The group D8 using the standard presentation
/// D8 = { 1, F, R | F^2 = R^4 = 1, FR = R^3F }
pub const D8: [D8Element; 8] = [
    // e
    D8Element([None; 4]),
    // F
    D8Element([Some(D8Generator::F), None, None, None]),
    // FR
    D8Element([Some(D8Generator::FR), None, None, None]),
    // FR^2
    D8Element([
        Some(D8Generator::FR),
        Some(D8Generator::F),
        Some(D8Generator::FR),
        None,
    ]),
    // FR^3
    D8Element([
        Some(D8Generator::F),
        Some(D8Generator::FR),
        Some(D8Generator::F),
        None,
    ]),
    // R
    D8Element([Some(D8Generator::FR), Some(D8Generator::F), None, None]),
    // R^2
    D8Element([
        Some(D8Generator::FR),
        Some(D8Generator::F),
        Some(D8Generator::FR),
        Some(D8Generator::F),
    ]),
    // R^3
    D8Element([Some(D8Generator::F), Some(D8Generator::FR), None, None]),
];
