use std::fmt;
use std::str::FromStr;

use anyhow::Context;
use serde::{Deserialize, Serialize};

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
    pub fn opposite(&self) -> Self {
        match self {
            Role::Attacker => Role::Defender,
            Role::Defender => Role::Attacker,
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
    #[must_use]
    pub fn fmt_other(&self) -> String {
        format!(
            "{}{}",
            BOARD_LETTERS.chars().collect::<Vec<_>>()[self.x],
            11 - self.y
        )
    }

    pub fn is_restricted(&self) -> bool {
        RESTRICTED_SQUARES.contains(&self)
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

    #[must_use]
    pub fn touches_wall(&self) -> bool {
        self.x == 0 || self.x == 10 || self.y == 0 || self.y == 10
    }

    pub fn iter() -> Self {
        Self { x: 0, y: 0 }
    }
}

impl Iterator for Square {
    type Item = Self;

    fn next(&mut self) -> Option<Self::Item> {
        if self.y < 10 {
            self.y += 1;
            Some(*self)
        } else if self.x == 10 {
            None
        } else {
            self.y = 0;
            self.x += 1;
            Some(*self)
        }
    }
}
