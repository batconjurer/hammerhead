use std::fmt;
use std::str::FromStr;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::game::Status;

pub const BOARD_LETTERS: &str = "ABCDEFGHIJK";

/// The two sides of the game
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum Role {
    #[default]
    Attacker,
    Defender,
}

impl Role {
    #[must_use]
    pub fn opposite(&self) -> Self {
        match self {
            Role::Attacker => Role::Defender,
            Role::Defender => Role::Attacker,
        }
    }

    pub fn victory(&self) -> Status {
        match self {
            Role::Attacker => Status::AttackersWin,
            Role::Defender => Status::DefendersWin,
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::Attacker => write!(f, "attacker"),
            Role::Defender => write!(f, "defender"),
        }
    }
}

impl FromStr for Role {
    type Err = anyhow::Error;

    fn from_str(string: &str) -> anyhow::Result<Self> {
        match string {
            "attacker" => Ok(Self::Attacker),
            "defender" => Ok(Self::Defender),
            _ => Err(anyhow::Error::msg(format!(
                "Error trying to convert '{string}' to a Role!"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum Space {
    Empty,
    Occupied(Role),
    King,
}

impl Space {
    /// Check if the piece occupying this square is on the
    /// same side as `role`.
    pub fn is_ally(&self, role: &Role) -> bool {
        match self {
            Space::Occupied(r) => r == role,
            Space::King => *role == Role::Defender,
            _ => false,
        }
    }
}

impl TryFrom<char> for Space {
    type Error = anyhow::Error;

    fn try_from(value: char) -> anyhow::Result<Self> {
        match value {
            'X' => Ok(Self::Occupied(Role::Defender)),
            'O' => Ok(Self::Occupied(Role::Attacker)),
            '.' => Ok(Self::Empty),
            'K' => Ok(Self::King),
            ch => Err(anyhow::Error::msg(format!(
                "Error trying to convert '{ch}' to a Space!"
            ))),
        }
    }
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Occupied(Role::Attacker) => write!(f, "♟"),
            Self::Empty => write!(f, "."),
            Self::King => write!(f, "♔"),
            Self::Occupied(Role::Defender) => write!(f, "♙"),
        }
    }
}

pub const EXIT_SQUARES: [Square; 4] = [
    Square { x: 0, y: 0 },
    Square { x: 10, y: 0 },
    Square { x: 0, y: 10 },
    Square { x: 10, y: 10 },
];

pub const THRONE: Square = Square { x: 5, y: 5 };

pub const RESTRICTED_SQUARES: [Square; 5] = [
    Square { x: 0, y: 0 },
    Square { x: 10, y: 0 },
    Square { x: 0, y: 10 },
    Square { x: 10, y: 10 },
    THRONE,
];

#[derive(Copy, Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Square {
    pub x: usize,
    pub y: usize,
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            BOARD_LETTERS.chars().collect::<Vec<_>>()[self.x],
            11 - self.y
        )
    }
}

impl FromStr for Square {
    type Err = anyhow::Error;

    fn from_str(vertex: &str) -> anyhow::Result<Self> {
        let mut chars = vertex.chars();

        if let Some(mut ch) = chars.next() {
            ch = ch.to_ascii_uppercase();
            let x = BOARD_LETTERS
                .find(ch)
                .context("play: the first letter is not a legal char")?;

            let mut y = chars.as_str().parse()?;
            if y > 0 && y < 12 {
                y = 11 - y;
                return Ok(Self { x, y });
            }
        }

        Err(anyhow::Error::msg("play: invalid coordinate"))
    }
}

impl Square {
    /// Checks if the square is one of the corners or the throne
    pub fn is_restricted(&self) -> bool {
        RESTRICTED_SQUARES.contains(self)
    }

    /// Checks if the square is one of the corners
    pub fn is_exit(&self) -> bool {
        EXIT_SQUARES.contains(self)
    }

    #[must_use]
    pub fn up(&self) -> Option<Square> {
        if self.y > 0 {
            Some(Square {
                x: self.x,
                y: self.y - 1,
            })
        } else {
            None
        }
    }

    #[must_use]
    pub fn left(&self) -> Option<Square> {
        if self.x > 0 {
            Some(Square {
                x: self.x - 1,
                y: self.y,
            })
        } else {
            None
        }
    }

    #[must_use]
    pub fn down(&self) -> Option<Square> {
        if self.y < 10 {
            Some(Square {
                x: self.x,
                y: self.y + 1,
            })
        } else {
            None
        }
    }

    #[must_use]
    pub fn right(&self) -> Option<Square> {
        if self.x < 10 {
            Some(Square {
                x: self.x + 1,
                y: self.y,
            })
        } else {
            None
        }
    }

    /// Get an iterator over all squares in the board
    pub fn iter() -> SquareIter {
        SquareIter::default()
    }
}

#[derive(Default)]
pub struct SquareIter(Option<Square>);
impl Iterator for SquareIter {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.as_mut() {
            None => self.0 = Some(Square { x: 0, y: 0 }),
            Some(sq) => {
                if sq.y < 10 {
                    sq.y += 1;
                } else if sq.x == 10 {
                    return None;
                } else {
                    sq.y = 0;
                    sq.x += 1;
                }
            }
        }
        self.0
    }
}

#[derive(Default)]
pub struct DefenderIter(Option<Square>);

impl Iterator for DefenderIter {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        const OUTER_LIMITS: [usize; 6] = [0, 1, 2, 8, 9, 10];
        match self.0.as_mut() {
            // Start out on the outer 3 layers
            None => self.0 = Some(Square { x: 3, y: 3 }),
            Some(sq) => {
                // if we are in the top or bottom 3 rows...
                if sq.y < 10 && OUTER_LIMITS.contains(&sq.x) {
                    sq.y += 1;
                } else if sq.y < 10 {
                    // if we are in the 3rd column, we just over the inner square
                    if sq.y == 2 {
                        sq.y = 8;
                    } else if sq.y == 7 {
                        // if we are in the 8th column and 8th row, we
                        // start iterating the outer layer
                        if sq.x == 7 {
                            sq.x = 0;
                            sq.y = 0;
                        } else {
                            // if we are in the 8th column, we are iterating over
                            // the inner square
                            sq.y = 3;
                            sq.x += 1;
                        }
                    } else {
                        // not in an edge condition, just advance
                        sq.y += 1;
                    }
                } else if sq.x == 10 {
                    // we have finished the outer layers, we are done
                    return None;
                } else {
                    // start over at the next row
                    sq.y = 0;
                    sq.x += 1;
                }
            }
        }
        self.0
    }
}

#[derive(Default)]
pub struct AttackerIter(Option<Square>);

impl Iterator for AttackerIter {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        const OUTER_LIMITS: [usize; 6] = [0, 1, 2, 8, 9, 10];
        match self.0.as_mut() {
            // Start out on the outer 3 layers
            None => self.0 = Some(Square { x: 0, y: 1 }),
            Some(sq) => {
                // if we are in the top or bottom 3 rows...
                if sq.y < 10 && OUTER_LIMITS.contains(&sq.x) {
                    sq.y += 1;
                } else if sq.y < 10 {
                    // if we are in the 3rd column, we just over the inner square
                    if sq.y == 2 {
                        sq.y = 8;
                    } else if sq.y == 7 {
                        // if we are in the 8th column and 8th row, we are done
                        if sq.x == 7 {
                            return None;
                        } else {
                            // if we are in the 8th column, we are iterating over
                            // the inner square
                            sq.y = 3;
                            sq.x += 1;
                        }
                    } else {
                        // not in an edge condition, just advance
                        sq.y += 1;
                    }
                } else if sq.x == 10 {
                    // we have finished the outer layers,
                    // now iterator over the inner square
                    sq.x = 3;
                    sq.y = 3;
                } else {
                    // start over at the next row
                    sq.y = 0;
                    sq.x += 1;
                }
            }
        }
        // we don't need to yield restricted squares
        if self.0.unwrap().is_restricted() {
            self.next()
        } else {
            self.0
        }
    }
}

#[derive(Debug)]
pub struct SquareMap<T> {
    inner: [Option<T>; 121],
}

impl<T> Default for SquareMap<T> {
    fn default() -> Self {
        Self {
            inner: [const { None }; 121],
        }
    }
}

impl<T> SquareMap<T> {
    pub fn contains_key(&self, key: &Square) -> bool {
        self.inner[key.y * 11 + key.x].is_some()
    }

    pub fn get(&self, key: &Square) -> Option<&T> {
        let ix = key.y * 11 + key.x;
        self.inner[ix].as_ref()
    }

    pub fn insert(&mut self, key: Square, value: T) {
        let ix = key.y * 11 + key.x;
        self.inner[ix] = Some(value);
    }
}

pub type SquareSet = SquareMap<()>;

impl SquareSet {
    pub fn contains(&self, key: &Square) -> bool {
        self.contains_key(key)
    }

    pub fn add(&mut self, key: Square) {
        self.insert(key, ());
    }
}

impl<A> FromIterator<(Square, A)> for SquareMap<A> {
    fn from_iter<T: IntoIterator<Item = (Square, A)>>(iter: T) -> Self {
        let mut map = SquareMap::default();
        for (key, value) in iter {
            map.insert(key, value);
        }
        map
    }
}

impl<A, T> From<A> for SquareMap<T>
where
    A: IntoIterator<Item = (Square, T)>,
{
    fn from(iter: A) -> Self {
        Self::from_iter(iter)
    }
}

#[cfg(test)]
mod test_spaces {
    use super::*;
    use std::collections::HashSet;

    /// Test that the iteration over the squares visits
    /// every square on the board exactly once.
    #[test]
    fn test_square_iter() {
        let mut squares = HashSet::new();
        for sq in Square::iter() {
            assert!(squares.insert(sq));
            assert!(sq.x < 11);
            assert!(sq.y < 11);
        }
        assert_eq!(squares.len(), 11 * 11);
    }

    /// Check that we can correctly tell which side a piece is on
    #[test]
    fn test_is_ally() {
        assert!(!Space::Empty.is_ally(&Role::Defender));
        assert!(!Space::Empty.is_ally(&Role::Attacker));
        assert!(Space::King.is_ally(&Role::Defender));
        assert!(!Space::King.is_ally(&Role::Attacker));
        assert!(Space::Occupied(Role::Defender).is_ally(&Role::Defender));
        assert!(!Space::Occupied(Role::Defender).is_ally(&Role::Attacker));
        assert!(Space::Occupied(Role::Attacker).is_ally(&Role::Attacker));
        assert!(!Space::Occupied(Role::Attacker).is_ally(&Role::Defender));
    }

    /// Spot checks that we label squares correctly
    #[test]
    fn test_fmt() {
        assert_eq!(Square { x: 0, y: 10 }.to_string(), "A1");
        assert_eq!(Square { x: 0, y: 0 }.to_string(), "A11");
        assert_eq!(Square { x: 4, y: 6 }.to_string(), "E5");
        assert_eq!(Square { x: 5, y: 5 }.to_string(), "F6");
    }

    /// Test that the [`AttackIter`] yeilds values in the outer
    /// three layers first and yields no restricted squares.
    #[test]
    fn test_attacker_iter() {
        const OUTER_VALUES: usize = 121 - 25 - 4;
        const INNER_VALUES: usize = 25 - 1;
        const OUTER_LIMITS: [usize; 6] = [0, 1, 2, 8, 9, 10];

        let mut iter = AttackerIter::default();
        let mut visited = HashSet::new();
        for _ in 0..OUTER_VALUES {
            let next = iter.next().expect("Test failed");
            assert!(OUTER_LIMITS.contains(&next.x) || OUTER_LIMITS.contains(&next.y));
            assert!(!next.is_restricted());
            assert!(visited.insert(next));
        }
        for _ in 0..INNER_VALUES {
            let next = iter.next().expect("Test failed");
            assert!(!OUTER_LIMITS.contains(&next.x) && !OUTER_LIMITS.contains(&next.y));
            assert!(!next.is_restricted());
            assert!(visited.insert(next));
        }
        assert_eq!(visited.len(), 121 - 5);
    }

    /// Test that the [`DefenderIter`] yields values in the inner 5 x 5
    /// square first.
    #[test]
    fn test_defender_iter() {
        const OUTER_VALUES: usize = 121 - 25;
        const INNER_VALUES: usize = 25;
        const OUTER_LIMITS: [usize; 6] = [0, 1, 2, 8, 9, 10];

        let mut iter = DefenderIter::default();
        let mut visited = HashSet::new();

        for _ in 0..INNER_VALUES {
            let next = iter.next().expect("Test failed");
            assert!(!OUTER_LIMITS.contains(&next.x) && !OUTER_LIMITS.contains(&next.y));
            assert!(visited.insert(next));
        }
        for _ in 0..OUTER_VALUES {
            let next = iter.next().expect("Test failed");
            assert!(OUTER_LIMITS.contains(&next.x) || OUTER_LIMITS.contains(&next.y));
            assert!(visited.insert(next));
        }

        assert_eq!(visited.len(), 121);
    }
}
