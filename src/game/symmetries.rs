//! Hnefatafl is symmetric with respect to the symmetries of the square,
//! the groupd D8. This contains utilities to exploit that symmetry.

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

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

/// For each symmetry of a board, compute a byte
/// vector. Return a hash of the sum of these vectors.
/// This provides a hash that is invariant under board symmetries
fn symmetric_hash(board: &Board) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut bytes = [[0u8; 30]; 8];
    for (ix, d8) in D8.iter().enumerate() {
        let mut b = board.clone();
        d8.apply(&mut b);
        bytes[ix] = b.as_bitboard();
    }
    bytes.sort_unstable();
    let mut hasher = Sha256::default();
    for b in bytes {
        hasher.update(b);
    }
    hasher.finalize().into()
}

/// A hash map for storing data about boards that are not affected
/// by the natural symmetries of the board.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NormalizedBoardMap<V>(FxHashMap<[u8; 32], V>);

impl<V> NormalizedBoardMap<V> {
    #[allow(dead_code)]
    pub fn insert(&mut self, board: &Board, value: V) -> Option<V> {
        self.0.insert(symmetric_hash(board), value)
    }

    #[allow(dead_code)]
    pub fn contains_key(&self, board: &Board) -> bool {
        self.0.contains_key(&symmetric_hash(board))
    }

    #[allow(dead_code)]
    pub fn remove(&mut self, board: &Board) -> Option<V> {
        self.0.remove(&symmetric_hash(board))
    }

    #[allow(dead_code)]
    pub fn get(&self, board: &Board) -> Option<&V> {
        self.0.get(&symmetric_hash(board))
    }

    #[allow(dead_code)]
    pub fn get_mut(&mut self, board: &Board) -> Option<&mut V> {
        self.0.get_mut(&symmetric_hash(board))
    }
}

/// A hash set version of [`NormalizedBoardMap`]
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NormalizedBoards(FxHashSet<[u8; 32]>);

impl NormalizedBoards {
    pub fn insert(&mut self, board: &Board) -> bool {
        self.0.insert(symmetric_hash(board))
    }

    pub fn contains(&self, board: &Board) -> bool {
        self.0.contains(&symmetric_hash(board))
    }

    pub fn remove(&mut self, board: &Board) -> bool {
        self.0.remove(&symmetric_hash(board))
    }
}

#[cfg(test)]
mod test_symmetries {
    use super::*;
    use crate::game::{Play, PositionsTracker, Status};
    use crate::game::space::Role;

    #[test]
    fn test_symmetric_hash() {
        let board = Board::try_from([
            "...OOOOO...",
            "...X....O..",
            ".........O.",
            "...O.X....O",
            "O....XX...O",
            "...O..XX..O",
            "O.O.....O.O",
            "OX.O.......",
            "..........K",
            ".....O.....",
            "....OO.O...",
        ])
        .expect("Test failed");

        let previous_boards = PositionsTracker::Counter(0);
        let mut normalized_boards = NormalizedBoards::default();
        for from in Square::iter() {
            for to in Square::iter() {
                let play = Play {
                    role: Role::Attacker,
                    from,
                    to,
                };
                if let Ok((board, _, _)) =
                    board.play_internal(&play, &Status::Ongoing, &previous_boards)
                {
                    assert!(normalized_boards.insert(&board))
                }
            }
        }
    }
}
