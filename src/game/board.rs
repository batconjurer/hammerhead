use crate::game::space::{
    BOARD_LETTERS, EXIT_SQUARES, RESTRICTED_SQUARES, Role, Space, Square, THRONE,
};
use crate::game::{Play, PreviousBoards, Status};
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::VecDeque;
use std::{collections::HashMap, fmt};

pub const STARTING_POSITION: [&str; 11] = [
    "...XXXXX...",
    ".....X.....",
    "...........",
    "X....O....X",
    "X...OOO...X",
    "XX.OOKOO.XX",
    "X...OOO...X",
    "X....O....X",
    "...........",
    ".....X.....",
    "...XXXXX...",
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
                match self.spaces[(y * 10) + x] {
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
    fn able_to_move(&self, play_from: &Square) -> bool {
        if let Some(square) = play_from.up() {
            if self.get(&square) == Space::Empty {
                return true;
            }
        }

        if let Some(square) = play_from.left() {
            if self.get(&square) == Space::Empty {
                return true;
            }
        }

        if let Some(square) = play_from.down() {
            if self.get(&square) == Space::Empty {
                return true;
            }
        }

        if let Some(square) = play_from.right() {
            if self.get(&square) == Space::Empty {
                return true;
            }
        }

        false
    }

    #[must_use]
    pub fn a_legal_move_exists(
        &self,
        status: &Status,
        turn: &Role,
        previous_boards: &PreviousBoards,
    ) -> bool {
        for src in Square::iter().filter(|sq| self.get(&sq).is_ally(turn)) {
            for dest in Square::iter() {
                let play = Play {
                    role: *turn,
                    from: src.clone(),
                    to: dest,
                };

                if self
                    .play_internal(&play, status, turn, previous_boards)
                    .is_ok()
                {
                    return true;
                }
            }
        }
        false
    }

    #[must_use]
    pub fn captured(&self) -> Captured {
        let mut defenders = 0;
        let mut attackers = 0;
        let mut king = true;

        for space in self.spaces {
            match space {
                Space::Occupied(Role::Defender) => defenders += 1,
                Space::Occupied(Role::Attacker) => attackers += 1,
                Space::Empty => {}
                Space::King => king = false,
            }
        }

        Captured {
            attackers: 24 - attackers,
            defenders: 12 - defenders,
            king,
        }
    }

    #[allow(clippy::collapsible_if)]
    fn captures(&mut self, dest: &Square, side: &Role) -> Vec<Square> {
        let mut captures = vec![];
        if let Some(up_1) = dest.up() {
            let space = self.get(&up_1);
            if space != Space::King && !space.is_ally(side) {
                if let Some(up_2) = up_1.up() {
                    if up_2.is_restricted() || self.get(&up_2).is_ally(side) {
                        self.set(&up_1, Space::Empty);
                        captures.push(up_1);
                    }
                }
            }
        }

        if let Some(left_1) = dest.left() {
            let space = self.get(&left_1);
            if space != Space::King && space.is_ally(side) {
                if let Some(left_2) = left_1.left() {
                    if left_2.is_restricted() || self.get(&left_2).is_ally(side) {
                        self.set(&left_1, Space::Empty);
                        captures.push(left_1);
                    }
                }
            }
        }

        if let Some(down_1) = dest.down() {
            let space = self.get(&down_1);
            if space != Space::King && space.is_ally(side) {
                if let Some(down_2) = down_1.down() {
                    if down_2.is_restricted() || self.get(&down_2).is_ally(side) {
                        self.set(&down_1, Space::Empty);
                        captures.push(down_1);
                    }
                }
            }
        }

        if let Some(right_1) = dest.right() {
            let space = self.get(&right_1);
            if space != Space::King && space.is_ally(side) {
                if let Some(right_2) = right_1.right() {
                    if right_2.is_restricted() || self.get(&right_2).is_ally(side) {
                        self.set(&right_1, Space::Empty);
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
            // not sandwiched between pieces of same side, no capture
            if space == Space::Empty {
                maybe_captured.clear();
                break;
            }
            // found a shield wall capture
            if space.is_ally(side) {
                break;
            }
            // still checking
            if self.get(&get_shield_pos(&sq)).is_ally(side) {
                maybe_captured.push(sq);
            } else {
                // not all piece are shielded against edge
                maybe_captured.clear();
                break;
            }
            next = get_next(&sq);
        }
        maybe_captured
    }

    fn captures_shield_wall(&mut self, side: &Role, dest: &Square) -> Vec<Square> {
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

    /// See if we can reach a defender from any corner by traversing through empty squares.
    /// If not, the attackers win.
    fn flood_fill_attackers_win(&self) -> bool {
        for corner in RESTRICTED_SQUARES.into_iter().filter(|sq| *sq != THRONE) {
            // Do a breadth-first search from the corner
            let mut queue = VecDeque::from([corner]);
            while let Some(sq) = queue.pop_front() {
                for neighbor in [sq.up(), sq.down(), sq.left(), sq.right()]
                    .into_iter()
                    .filter_map(|x| x)
                {
                    match self.get(&neighbor) {
                        // if we can reach a defender, the attackers have not won
                        Space::Occupied(Role::Defender) | Space::King => return false,
                        // we cannot pass through attackers
                        Space::Occupied(Role::Attacker) => {}
                        // traverse through this space
                        Space::Empty => queue.push_back(neighbor),
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

    #[must_use]
    fn no_attackers_left(&self) -> bool {
        for sq in Square::iter() {
            if self.get(&sq) == Space::Occupied(Role::Attacker) {
                return false;
            }
        }
        true
    }

    /// # Errors
    ///
    /// If the vertex is out of bounds.
    pub fn play(
        &mut self,
        play: &Play,
        status: &Status,
        turn: &Role,
        previous_boards: &mut PreviousBoards,
    ) -> anyhow::Result<(Vec<Square>, Status)> {
        let (board, captures, status) = self.play_internal(play, status, turn, previous_boards)?;
        previous_boards.0.insert(board.clone());
        *self = board;

        Ok((captures, status))
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::missing_errors_doc
    )]
    pub fn play_internal(
        &self,
        play: &Play,
        status: &Status,
        turn: &Role,
        previous_boards: &PreviousBoards,
    ) -> anyhow::Result<(Board, Vec<Square>, Status)> {
        if *status != Status::Ongoing {
            return Err(anyhow::Error::msg(
                "play: the game has to be ongoing to play",
            ));
        }
        play.valid()?;

        let space_from = self.get(&play.from);
        if !space_from.is_ally(turn) {
            return Err(anyhow::Error::msg(format!(
                "play: you must select of piece of type {turn}"
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

        if previous_boards.0.contains(&board) && turn == &Role::Defender {
            return Err(anyhow::Error::msg(
                "play: you already reached that position",
            ));
        }

        let mut captures = Vec::new();
        captures.extend(board.captures(&play.to, turn));
        captures.extend(board.captures_shield_wall(turn, &play.to));

        if EXIT_SQUARES.contains(&play.to) {
            return Ok((board, captures, Status::DefendersWin));
        }

        if board.capture_the_king() {
            return Ok((board, captures, Status::AttackersWin));
        }

        if board.flood_fill_attackers_win() {
            return Ok((board, captures, Status::AttackersWin));
        }

        if board.no_attackers_left() {
            return Ok((board, captures, Status::DefendersWin));
        }

        if previous_boards.0.len() > 100 {
            return Ok((board, captures, Status::AttackersWin));
        }

        Ok((board, captures, Status::Ongoing))
    }

    fn set(&mut self, square: &Square, space: Space) {
        self.spaces[square.y * 11 + square.x] = space;
    }

    #[must_use]
    fn set_if_not_king(&mut self, square: &Square, space: Space) -> bool {
        if self.get(square) == Space::King {
            false
        } else {
            self.set(square, space);
            true
        }
    }
}

pub struct Captured {
    defenders: u8,
    attackers: u8,
    king: bool,
}

impl Captured {
    #[must_use]
    pub fn defender(&self) -> String {
        let mut string = format!("♟ {}", self.defenders);
        if self.king {
            string.push_str(" ♔");
        }
        string
    }

    #[must_use]
    pub fn attacker(&self) -> String {
        format!("♙ {}", self.attackers)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Direction {
    LeftRight,
    UpDown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegalMoves {
    pub role: Role,
    pub moves: HashMap<Square, Vec<Square>>,
}

#[must_use]
fn expand_flood_fill(
    sq: Option<Square>,
    already_checked: &mut FxHashSet<Square>,
    stack: &mut Vec<Square>,
) -> bool {
    if let Some(sq) = sq {
        if !already_checked.contains(&sq) {
            stack.push(sq.clone());
            already_checked.insert(sq);
        }
        true
    } else {
        false
    }
}
