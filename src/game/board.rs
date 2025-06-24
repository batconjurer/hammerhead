use std::collections::{HashSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::game::space::{
    BOARD_LETTERS, EXIT_SQUARES, RESTRICTED_SQUARES, Role, Space, Square, THRONE,
};
use crate::game::{Play, PreviousBoards, Status};

pub const STARTING_POSITION: [&str; 11] = [
    "...OOOOO...",
    ".....O.....",
    "...........",
    "O....X....O",
    "O...XXX...O",
    "OO.XXKXX.OO",
    "O...XXX...O",
    "O....X....O",
    "...........",
    ".....O.....",
    "...OOOOO...",
];

#[serde_as]
#[derive(Clone, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Board {
    #[serde_as(as = "[_; 121]")]
    pub spaces: [Space; 11 * 11],
}

impl Default for Board {
    fn default() -> Self {
        STARTING_POSITION.try_into().unwrap()
    }
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        for y in 0..11 {
            write!(f, r#"""#)?;

            for x in 0..11 {
                match self.spaces[(y * 11) + x] {
                    Space::Occupied(Role::Defender) => write!(f, "X")?,
                    Space::Empty => write!(f, ".")?,
                    Space::King => write!(f, "K")?,
                    Space::Occupied(Role::Attacker) => write!(f, "O")?,
                }
            }
            writeln!(f, r#"""#)?;
        }

        Ok(())
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut letters = " ".repeat(3).to_string();
        letters.push_str(BOARD_LETTERS);
        let bar = "─".repeat(11);

        writeln!(f, "\n{letters}\n  ┌{bar}┐")?;
        for y in 0..11 {
            let y_label = 11 - y;
            write!(f, "{y_label:2}│",)?;

            for x in 0..11 {
                if ((y, x) == (0, 0)
                    || (y, x) == (10, 0)
                    || (y, x) == (0, 10)
                    || (y, x) == (10, 10)
                    || (y, x) == (5, 5))
                    && self.spaces[y * 11 + x] == Space::Empty
                {
                    write!(f, "⌘")?;
                } else {
                    write!(f, "{}", self.spaces[y * 11 + x])?;
                }
            }
            writeln!(f, "│{y_label:2}")?;
        }
        write!(f, "  └{bar}┘\n{letters}")
    }
}

impl TryFrom<[&str; 11]> for Board {
    type Error = anyhow::Error;

    fn try_from(value: [&str; 11]) -> anyhow::Result<Self> {
        let mut spaces = [Space::Empty; 11 * 11];
        let mut kings = 0;

        for (y, row) in value.iter().enumerate() {
            for (x, ch) in row.chars().enumerate() {
                let space = ch.try_into()?;
                match space {
                    Space::Occupied(_) => {
                        let vertex = Square { x, y };
                        if RESTRICTED_SQUARES.contains(&vertex) {
                            return Err(anyhow::Error::msg(
                                "Only the king is allowed on restricted squares!",
                            ));
                        }
                    }
                    Space::Empty => {}
                    Space::King => {
                        kings += 1;
                        if kings > 1 {
                            return Err(anyhow::Error::msg("You can only have one king!"));
                        }
                    }
                }

                spaces[y * 11 + x] = space;
            }
        }

        Ok(Self { spaces })
    }
}

impl Board {
    /// Check if a given player can make a legal move
    #[must_use]
    pub fn a_legal_move_exists(&self, turn: &Role) -> bool {
        for src in Square::iter().filter(|sq| self.get(sq).is_ally(turn)) {
            for dest in [src.left(), src.right(), src.up(), src.down()]
                .into_iter()
                .flatten()
            {
                if !dest.is_restricted() && self.get(&dest) == Space::Empty {
                    return true;
                }
            }
        }
        false
    }

    /// Find which non-King pieces are captured when player `side` moves
    /// to square `dest`.
    #[allow(clippy::collapsible_if)]
    fn captures(&self, dest: &Square, side: &Role) -> Vec<Square> {
        let mut captures = vec![];
        if let Some(up_1) = dest.up() {
            let space = self.get(&up_1);
            if space != Space::King && space != Space::Empty && !space.is_ally(side) {
                if let Some(up_2) = up_1.up() {
                    if up_2.is_restricted() || self.get(&up_2).is_ally(side) {
                        captures.push(up_1);
                    }
                }
            }
        }

        if let Some(left_1) = dest.left() {
            let space = self.get(&left_1);
            if space != Space::King && space != Space::Empty && !space.is_ally(side) {
                if let Some(left_2) = left_1.left() {
                    if left_2.is_restricted() || self.get(&left_2).is_ally(side) {
                        captures.push(left_1);
                    }
                }
            }
        }

        if let Some(down_1) = dest.down() {
            let space = self.get(&down_1);
            if space != Space::King && space != Space::Empty && !space.is_ally(side) {
                if let Some(down_2) = down_1.down() {
                    if down_2.is_restricted() || self.get(&down_2).is_ally(side) {
                        captures.push(down_1);
                    }
                }
            }
        }

        if let Some(right_1) = dest.right() {
            let space = self.get(&right_1);
            if space != Space::King && space != Space::Empty && !space.is_ally(side) {
                if let Some(right_2) = right_1.right() {
                    if right_2.is_restricted() || self.get(&right_2).is_ally(side) {
                        captures.push(right_1);
                    }
                }
            }
        }
        captures
    }

    /// Check for a shield wall capture starting with piece `dest` on side
    /// `side`. The provided closures dictate in which direction and on which
    /// edge to check.
    fn shield_wall_aux<F, G>(
        &self,
        dest: &Square,
        side: &Role,
        get_next: F,
        get_shield_pos: G,
    ) -> Vec<Square>
    where
        F: Fn(&Square) -> Option<Square>,
        G: Fn(&Square) -> Square,
    {
        let mut next = get_next(dest);
        let mut maybe_captured = Vec::with_capacity(11);
        while let Some(sq) = next {
            let space = self.get(&sq);

            // found a shield wall capture
            if space.is_ally(side) || sq.is_restricted() {
                break;
            }

            // not sandwiched between pieces of same side, no capture
            if space == Space::Empty {
                maybe_captured.clear();
                break;
            }
            // still checking
            if self.get(&get_shield_pos(&sq)).is_ally(side) {
                // kings are not captured by shield walls
                if space != Space::King {
                    maybe_captured.push(sq);
                }
            } else {
                // not all piece are shielded against edge
                maybe_captured.clear();
                break;
            }
            next = get_next(&sq);
        }
        maybe_captured
    }

    /// Determine if a shield wall capture occurs when player `side` moves a piece
    /// to square `dest`.
    pub fn captures_shield_wall(&self, side: &Role, dest: &Square) -> Vec<Square> {
        let mut captures = Vec::with_capacity(22);
        if dest.x == 0 {
            captures.extend(self.shield_wall_aux(
                dest,
                side,
                |sq| sq.up(),
                |sq| sq.right().unwrap(),
            ));
            captures.extend(self.shield_wall_aux(
                dest,
                side,
                |sq| sq.down(),
                |sq| sq.right().unwrap(),
            ));
        }
        if dest.x == 10 {
            captures.extend(self.shield_wall_aux(
                dest,
                side,
                |sq| sq.up(),
                |sq| sq.left().unwrap(),
            ));
            captures.extend(self.shield_wall_aux(
                dest,
                side,
                |sq| sq.down(),
                |sq| sq.left().unwrap(),
            ));
        }

        if dest.y == 0 {
            captures.extend(self.shield_wall_aux(
                dest,
                side,
                |sq| sq.left(),
                |sq| sq.down().unwrap(),
            ));
            captures.extend(self.shield_wall_aux(
                dest,
                side,
                |sq| sq.right(),
                |sq| sq.down().unwrap(),
            ));
        }

        if dest.y == 10 {
            captures.extend(self.shield_wall_aux(
                dest,
                side,
                |sq| sq.left(),
                |sq| sq.up().unwrap(),
            ));
            captures.extend(self.shield_wall_aux(
                dest,
                side,
                |sq| sq.right(),
                |sq| sq.up().unwrap(),
            ));
        }
        captures
    }

    /// If the king is not captured, find the square on which he is located.
    pub fn find_the_king(&self) -> Option<Square> {
        self.spaces
            .iter()
            .enumerate()
            .find(|(_, s)| matches!(s, Space::King))
            .map(|(ix, _)| Square {
                x: ix.rem_euclid(11),
                y: ix / 11,
            })
    }

    /// Determine if the king is surrounded on all four sides by attackers
    fn capture_the_king(&self) -> bool {
        match self.find_the_king() {
            Some(king) => {
                for sq in [king.up(), king.down(), king.left(), king.right()] {
                    if let Some(sq) = sq.as_ref() {
                        let space = self.get(sq);
                        if space.is_ally(&Role::Defender) || space == Space::Empty {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// A corner case of a blocked corner that the flood fill algorithm
    /// doesn't handle correctly
    fn special_corner_block(&self, corner: &Square) -> bool {
        let to_check = match corner {
            Square { x: 0, y: 0 } => [
                Square { x: 1, y: 0 },
                Square { x: 2, y: 0 },
                Square { x: 0, y: 1 },
                Square { x: 0, y: 2 },
            ],
            Square { x: 0, y: 10 } => [
                Square { x: 1, y: 10 },
                Square { x: 2, y: 10 },
                Square { x: 0, y: 9 },
                Square { x: 0, y: 8 },
            ],
            Square { x: 10, y: 0 } => [
                Square { x: 9, y: 0 },
                Square { x: 8, y: 0 },
                Square { x: 10, y: 1 },
                Square { x: 10, y: 2 },
            ],
            Square { x: 10, y: 10 } => [
                Square { x: 9, y: 10 },
                Square { x: 8, y: 10 },
                Square { x: 10, y: 9 },
                Square { x: 10, y: 8 },
            ],
            _ => unreachable!(),
        };
        for sq in to_check {
            if !self.get(&sq).is_ally(&Role::Attacker) {
                return false;
            }
        }
        true
    }

    /// See if we can reach a defender from any corner by traversing through empty squares.
    /// If not, the attackers win.
    ///
    /// N.B. There are rare cases where a corner is blocked with an attacker sandwiched
    /// inside. This algorithm will not detect this.
    fn flood_fill_attackers_win(&self) -> bool {
        for corner in RESTRICTED_SQUARES.into_iter().filter(|sq| *sq != THRONE) {
            if self.special_corner_block(&corner) {
                continue;
            }
            // Do a breadth-first search from the corner
            let mut queue = VecDeque::from([corner]);
            let mut visited = HashSet::from([corner]);
            while let Some(sq) = queue.pop_front() {
                for neighbor in [sq.up(), sq.down(), sq.left(), sq.right()]
                    .into_iter()
                    .flatten()
                {
                    match self.get(&neighbor) {
                        // if we can reach a defender, the attackers have not won
                        Space::Occupied(Role::Defender) | Space::King => return false,
                        // we cannot pass through attackers unless we are at the corner
                        Space::Occupied(Role::Attacker) => {
                            if sq == corner && !visited.contains(&neighbor) {
                                queue.push_back(neighbor);
                                visited.insert(neighbor);
                            }
                        }
                        // traverse through this space
                        Space::Empty => {
                            if !visited.contains(&neighbor) {
                                queue.push_back(neighbor);
                                visited.insert(neighbor);
                            }
                        }
                    }
                }
            }
        }
        true
    }

    #[must_use]
    pub fn get(&self, square: &Square) -> Space {
        self.spaces[square.y * 11 + square.x]
    }

    /// Play a move. Errors if the play is invalid or the game is already over.
    /// Stores the board in the history for checking repeated positions and enforcing
    /// the one hundred move limit.
    pub fn play(
        &mut self,
        play: &Play,
        status: &Status,
        previous_boards: &mut PreviousBoards,
    ) -> anyhow::Result<(Vec<Square>, Status)> {
        let (board, captures, status) = self.play_internal(play, status, previous_boards)?;
        previous_boards.0.insert(board.clone());
        *self = board;

        Ok((captures, status))
    }

    /// The actual game logic. Checks if a move is valid, computes
    /// captures, and checks if the game is won.
    ///
    /// Errors on an illegal move
    pub fn play_internal(
        &self,
        play: &Play,
        status: &Status,
        previous_boards: &PreviousBoards,
    ) -> anyhow::Result<(Board, Vec<Square>, Status)> {
        if *status != Status::Ongoing {
            return Err(anyhow::Error::msg(
                "play: the game has to be ongoing to play",
            ));
        }
        play.valid()?;

        let space_from = self.get(&play.from);
        if !space_from.is_ally(&play.role) {
            return Err(anyhow::Error::msg(format!(
                "play: you must select of piece of type {}",
                play.role,
            )));
        }

        let x_diff = play.from.x as i32 - play.to.x as i32;
        let y_diff = play.from.y as i32 - play.to.y as i32;

        if x_diff != 0 {
            let x_diff_sign = x_diff.signum();
            for x_diff in 1..=x_diff.abs() {
                let sq = Square {
                    x: (play.from.x as i32 - (x_diff * x_diff_sign)) as usize,
                    y: play.from.y,
                };

                let space = self.get(&sq);
                if space != Space::Empty {
                    return Err(anyhow::Error::msg(
                        "play: pieces may not move through other pieces",
                    ));
                }
            }
        } else {
            let y_diff_sign = y_diff.signum();
            for y_diff in 1..=y_diff.abs() {
                let sq = Square {
                    x: play.from.x,
                    y: (play.from.y as i32 - (y_diff * y_diff_sign)) as usize,
                };
                let space = self.get(&sq);
                if space != Space::Empty {
                    return Err(anyhow::Error::msg(
                        "play: pieces may not move through other pieces",
                    ));
                }
            }
        }

        if space_from != Space::King && RESTRICTED_SQUARES.contains(&play.to) {
            return Err(anyhow::Error::msg(
                "play: only the king may move to a restricted square",
            ));
        }

        let mut board = self.clone();
        board.set(&play.from, Space::Empty);
        board.set(&play.to, space_from);

        let mut captures = Vec::new();
        captures.extend(board.captures(&play.to, &play.role));
        captures.extend(board.captures_shield_wall(&play.role, &play.to));
        for capture in &captures {
            board.set(capture, Space::Empty);
        }

        if EXIT_SQUARES.contains(&play.to) {
            return Ok((board, captures, Status::DefendersWin));
        }

        if board.capture_the_king() {
            return Ok((board, captures, Status::AttackersWin));
        }

        if previous_boards.0.contains(&board) && play.role == Role::Defender {
            return Ok((board, captures, Status::AttackersWin));
        }

        if board.flood_fill_attackers_win() {
            return Ok((board, captures, Status::AttackersWin));
        }

        if !board.a_legal_move_exists(&play.role.opposite()) {
            return Ok((board, captures, play.role.victory()));
        }

        if previous_boards.0.len() >= 100 {
            return Ok((board, captures, Status::Draw));
        }

        Ok((board, captures, Status::Ongoing))
    }

    fn set(&mut self, square: &Square, space: Space) {
        self.spaces[square.y * 11 + square.x] = space;
    }
}

#[cfg(test)]
mod test_board {
    use super::*;

    /// Test we can detect if a side still has a legal move
    #[test]
    fn test_legal_move_exists() {
        let board = Board::default();
        assert!(board.a_legal_move_exists(&Role::Defender));
        assert!(board.a_legal_move_exists(&Role::Attacker));
        let board = [
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            ".........O.",
            "........OX.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(!board.a_legal_move_exists(&Role::Defender));
        assert!(board.a_legal_move_exists(&Role::Attacker));
        let board = [
            "...........",
            "...........",
            "...........",
            "...........",
            "...X.X.....",
            "..XOXKX....",
            "...X.X.....",
            "...........",
            "...........",
            ".........X.",
            "........XO.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(board.a_legal_move_exists(&Role::Defender));
        assert!(!board.a_legal_move_exists(&Role::Attacker));
    }

    /// Test that captured pieces are correctly computed
    #[test]
    fn test_captures() {
        // check that the corner partakes in captures
        let board = [
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "......OXOX.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert_eq!(
            board.captures(&Square { x: 8, y: 10 }, &Role::Attacker),
            vec![Square { x: 7, y: 10 }, Square { x: 9, y: 10 }]
        );
        // check that we don't compute king captures with this method
        // and we don't capture empty spaces
        let board = [
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            ".....O.....",
            ".....X.....",
            "...OXO.O...",
            "....OKO....",
            ".....O.....",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert_eq!(
            board.captures(&Square { x: 5, y: 8 }, &Role::Attacker),
            vec![Square { x: 5, y: 7 }, Square { x: 4, y: 8 }]
        );
        // check we don't capture allies
        let board = [
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            ".....O.....",
            ".....O.....",
            "...OXO.O...",
            "....OKO....",
            ".....O.....",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert_eq!(
            board.captures(&Square { x: 5, y: 8 }, &Role::Attacker),
            vec![Square { x: 4, y: 8 }]
        );
    }

    /// Check that we correctly identify shield wall captures
    #[test]
    fn test_shield_walls() {
        // valid shield wall with two capture sets,
        // one of which includes the king, and using
        // a corner as a flanking piece
        let board = [
            "...........",
            "O..........",
            "XO.........",
            "XO.........",
            "XO.........",
            "XO.........",
            "O..........",
            "XO.........",
            "KO.........",
            "XO.........",
            "...........",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let captures = board.captures_shield_wall(&Role::Attacker, &Square { x: 0, y: 6 });
        assert_eq!(
            captures,
            vec![
                Square { x: 0, y: 5 },
                Square { x: 0, y: 4 },
                Square { x: 0, y: 3 },
                Square { x: 0, y: 2 },
                Square { x: 0, y: 7 },
                Square { x: 0, y: 9 },
            ],
        );
        // non-flanking moves should not result in shield captures
        let captures = board.captures_shield_wall(&Role::Attacker, &Square { x: 1, y: 5 });
        assert!(captures.is_empty());
        // the situation on the left should be ignored
        let board = [
            "...........",
            "O.........O",
            "XO.......OX",
            "XO.......OX",
            "XO.......OX",
            "XO.......OX",
            "O.........O",
            "XO.......OX",
            "XO.......OK",
            "O.........O",
            "...........",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let captures = board.captures_shield_wall(&Role::Attacker, &Square { x: 10, y: 6 });
        assert_eq!(
            captures,
            vec![
                Square { x: 10, y: 5 },
                Square { x: 10, y: 4 },
                Square { x: 10, y: 3 },
                Square { x: 10, y: 2 },
                Square { x: 10, y: 7 },
            ],
        );
        let board = [
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            ".OOO..O....",
            "..KXOXXO...",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let captures = board.captures_shield_wall(&Role::Attacker, &Square { x: 4, y: 10 });
        assert!(captures.is_empty());
    }

    /// Test we detect when capturing the king
    #[test]
    fn test_king_capture() {
        // no king
        let board = [
            "...........",
            "O..........",
            "XO.........",
            "XO.........",
            "XO.........",
            "XO.........",
            "O..........",
            "XO.........",
            ".O.........",
            "XO.........",
            "...........",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(!board.capture_the_king());
        // not a king capture
        let board = [
            "...........",
            "O..........",
            "XO.........",
            "XO.........",
            "XO.........",
            "XO.........",
            "...........",
            "OO.........",
            "KO.........",
            "O..........",
            "...........",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(!board.capture_the_king());
        // throne does not partake in capture
        let board = [
            "...........",
            "...........",
            "...........",
            "...........",
            "....O......",
            "...OK......",
            "....O......",
            "...........",
            "...........",
            "...........",
            "...........",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(!board.capture_the_king());
        // a real king capture
        let board = [
            "...........",
            "...........",
            "...........",
            "...........",
            "...OO......",
            "..OKO......",
            "...OO......",
            "...........",
            "...........",
            "...........",
            "...........",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(board.capture_the_king());
    }

    #[test]
    fn test_special_corner_block() {
        let board = [
            ".OO.....OO.",
            "O........OO",
            "O.........O",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "O........OO",
            "O........OO",
            ".OO.....OO.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(board.special_corner_block(&Square { x: 0, y: 0 }));
        assert!(board.special_corner_block(&Square { x: 0, y: 10 }));
        assert!(board.special_corner_block(&Square { x: 10, y: 0 }));
        assert!(board.special_corner_block(&Square { x: 10, y: 10 }));
        let board = [
            ".OO.....O..",
            "O........OO",
            "..........O",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "X..........",
            "O........OO",
            ".OX.....OO.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(!board.special_corner_block(&Square { x: 0, y: 0 }));
        assert!(!board.special_corner_block(&Square { x: 0, y: 10 }));
        assert!(!board.special_corner_block(&Square { x: 10, y: 0 }));
        assert!(!board.special_corner_block(&Square { x: 10, y: 10 }));
    }

    /// Test that if the attackers block the corners,
    /// they win
    #[test]
    fn test_attackers_win_flood_fill() {
        let board = Board::default();
        assert!(!board.flood_fill_attackers_win());
        let board = [
            ".O......O..",
            "O........O.",
            "..........O",
            "...........",
            "...........",
            "X..........",
            "...........",
            "...........",
            "O........OO",
            "O........O.",
            ".OO.....O..",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(!board.flood_fill_attackers_win());
        let board = [
            "..O.....O..",
            "OOO......O.",
            "O.........O",
            "...........",
            "...........",
            "X..........",
            "...........",
            "...........",
            "O........OO",
            "O........O.",
            ".OO.....O..",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(board.flood_fill_attackers_win());
        let board = [
            "..O....OXO.",
            "OOO.....OO.",
            "O.........O",
            "...........",
            "...........",
            "X..........",
            "...........",
            "...........",
            "O........OO",
            "O........O.",
            ".OO.....O..",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(!board.flood_fill_attackers_win());
    }

    /// Test that moving an opponents piece is forbidden
    #[test]
    fn test_move_opponents_piece() {
        let board = Board::default();
        let err = board
            .play_internal(
                &Play {
                    role: Role::Defender,
                    from: Square { x: 3, y: 10 },
                    to: Square { x: 3, y: 9 },
                },
                &Status::Ongoing,
                &Default::default(),
            )
            .unwrap_err()
            .to_string();
        assert_eq!(err, "play: you must select of piece of type defender");

        let err = board
            .play_internal(
                &Play {
                    role: Role::Attacker,
                    from: Square { x: 5, y: 3 },
                    to: Square { x: 5, y: 4 },
                },
                &Status::Ongoing,
                &Default::default(),
            )
            .unwrap_err()
            .to_string();
        assert_eq!(err, "play: you must select of piece of type attacker");
    }

    /// Test that moving pieces through other pieces is forbidden
    #[test]
    fn test_moving_through_other_pieces() {
        let board = Board::default();
        let err = board
            .play_internal(
                &Play {
                    role: Role::Defender,
                    from: THRONE,
                    to: Square { x: 2, y: 5 },
                },
                &Status::Ongoing,
                &Default::default(),
            )
            .unwrap_err()
            .to_string();
        assert_eq!(err, "play: pieces may not move through other pieces");
        let err = board
            .play_internal(
                &Play {
                    role: Role::Defender,
                    from: THRONE,
                    to: Square { x: 8, y: 5 },
                },
                &Status::Ongoing,
                &Default::default(),
            )
            .unwrap_err()
            .to_string();
        assert_eq!(err, "play: pieces may not move through other pieces");
        let err = board
            .play_internal(
                &Play {
                    role: Role::Defender,
                    from: THRONE,
                    to: Square { x: 5, y: 8 },
                },
                &Status::Ongoing,
                &Default::default(),
            )
            .unwrap_err()
            .to_string();
        assert_eq!(err, "play: pieces may not move through other pieces");
        let err = board
            .play_internal(
                &Play {
                    role: Role::Defender,
                    from: THRONE,
                    to: Square { x: 5, y: 2 },
                },
                &Status::Ongoing,
                &Default::default(),
            )
            .unwrap_err()
            .to_string();
        assert_eq!(err, "play: pieces may not move through other pieces");
    }

    /// Test moving to / through restricted squares
    #[test]
    fn test_restricted_squares() {
        let board = [
            ".....O.....",
            "...........",
            "...........",
            "...........",
            "...........",
            "K..........",
            "...........",
            "...........",
            "...........",
            "..........X",
            ".....X.....",
        ];

        let board = Board::try_from(board).expect("Test failed");
        // passing through throne is allowed
        assert!(
            board
                .play_internal(
                    &Play {
                        role: Role::Attacker,
                        from: Square { x: 5, y: 0 },
                        to: Square { x: 5, y: 9 },
                    },
                    &Status::Ongoing,
                    &Default::default(),
                )
                .is_ok()
        );
        assert!(
            board
                .play_internal(
                    &Play {
                        role: Role::Defender,
                        from: Square { x: 5, y: 10 },
                        to: Square { x: 5, y: 1 },
                    },
                    &Status::Ongoing,
                    &Default::default(),
                )
                .is_ok()
        );
        // stopping on throne is forbidden
        let err = board
            .play_internal(
                &Play {
                    role: Role::Attacker,
                    from: Square { x: 5, y: 0 },
                    to: Square { x: 5, y: 5 },
                },
                &Status::Ongoing,
                &Default::default(),
            )
            .unwrap_err()
            .to_string();
        assert_eq!(err, "play: only the king may move to a restricted square");
        let err = board
            .play_internal(
                &Play {
                    role: Role::Defender,
                    from: Square { x: 5, y: 10 },
                    to: Square { x: 5, y: 5 },
                },
                &Status::Ongoing,
                &Default::default(),
            )
            .unwrap_err()
            .to_string();
        assert_eq!(err, "play: only the king may move to a restricted square");
        // only king can move to corner
        let err = board
            .play_internal(
                &Play {
                    role: Role::Attacker,
                    from: Square { x: 5, y: 0 },
                    to: Square { x: 10, y: 0 },
                },
                &Status::Ongoing,
                &Default::default(),
            )
            .unwrap_err()
            .to_string();
        assert_eq!(err, "play: only the king may move to a restricted square");
        let err = board
            .play_internal(
                &Play {
                    role: Role::Defender,
                    from: Square { x: 10, y: 9 },
                    to: Square { x: 10, y: 10 },
                },
                &Status::Ongoing,
                &Default::default(),
            )
            .unwrap_err()
            .to_string();
        assert_eq!(err, "play: only the king may move to a restricted square");
        // king can move to restricted squares
        assert!(
            board
                .play_internal(
                    &Play {
                        role: Role::Defender,
                        from: Square { x: 0, y: 5 },
                        to: Square { x: 5, y: 5 },
                    },
                    &Status::Ongoing,
                    &Default::default(),
                )
                .is_ok()
        );
        assert!(
            board
                .play_internal(
                    &Play {
                        role: Role::Defender,
                        from: Square { x: 0, y: 5 },
                        to: Square { x: 0, y: 10 },
                    },
                    &Status::Ongoing,
                    &Default::default(),
                )
                .is_ok()
        );
    }

    /// Test rules regarding repetitions of board positions
    #[test]
    fn test_repetitions() {
        let board = [
            "...OOOOO...",
            ".....O.....",
            "...........",
            "O....X....O",
            "O...XXX...O",
            "OO..XKXX.OO",
            "O..XXXX...O",
            "O....X....O",
            "...........",
            ".....O.....",
            "...OOOOO...",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let mut previous_boards = PreviousBoards::default();
        previous_boards.0.insert(Board::default());
        // cannot repeat if defender
        let (_, _, status) = board
            .play_internal(
                &Play {
                    role: Role::Defender,
                    from: Square { x: 3, y: 6 },
                    to: Square { x: 3, y: 5 },
                },
                &Status::Ongoing,
                &previous_boards,
            )
            .expect("Test failed");
        assert_eq!(status, Status::AttackersWin);
        // can repeat if attacker
        let board = [
            "...OOOOO...",
            ".....O.....",
            "...........",
            "O....X....O",
            "O...XXX...O",
            "O..XXKXX.OO",
            "O...XXX...O",
            "O....X....O",
            "...........",
            ".O...O.....",
            "...OOOOO...",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(
            board
                .play_internal(
                    &Play {
                        role: Role::Attacker,
                        from: Square { x: 1, y: 9 },
                        to: Square { x: 1, y: 5 },
                    },
                    &Status::Ongoing,
                    &previous_boards,
                )
                .is_ok()
        );
    }

    /// Test that defenders win if the king reaches a corner
    #[test]
    fn test_king_escape() {
        let board = [
            "...OOOOO...",
            ".....O.....",
            "...........",
            "O....X....O",
            "O...XXX...O",
            "OO..X.XX.OO",
            "O..XXXX...O",
            "O....X....O",
            "..........K",
            ".....O.....",
            "...OOOOO...",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let (_, _, status) = board
            .play_internal(
                &Play {
                    role: Role::Defender,
                    from: Square { x: 10, y: 8 },
                    to: Square { x: 10, y: 10 },
                },
                &Status::Ongoing,
                &Default::default(),
            )
            .expect("Test failed");
        assert_eq!(status, Status::DefendersWin);
    }

    #[test]
    fn test_captures_removes_pieces() {
        let board = [
            "...OOOOO...",
            ".....OX....",
            "...........",
            "O....X....O",
            "O...XX....O",
            "OO.XXKXX.OO",
            "O...XXX...O",
            "O....X....O",
            "...........",
            ".....O.....",
            "...OOOOO...",
        ];
        let mut board = Board::try_from(board).expect("Test failed");
        let mut previous_boards = Default::default();
        board
            .play(
                &Play {
                    role: Role::Attacker,
                    from: Square { x: 7, y: 0 },
                    to: Square { x: 7, y: 1 },
                },
                &Status::Ongoing,
                &mut previous_boards,
            )
            .expect("Test failed");
        let board_after = [
            "...OOOO....",
            ".....O.O...",
            "...........",
            "O....X....O",
            "O...XX....O",
            "OO.XXKXX.OO",
            "O...XXX...O",
            "O....X....O",
            "...........",
            ".....O.....",
            "...OOOOO...",
        ];
        let board_after = Board::try_from(board_after).expect("Test failed");
        assert_eq!(board, board_after);
    }
}
