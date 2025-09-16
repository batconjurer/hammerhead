use std::collections::HashSet;
use std::fmt::{Debug, Formatter};

use crate::game::board::Board;
use crate::game::{Play, PreviousBoards, Status};
use crate::game::space::{EXIT_SQUARES, Role, Square, SquareIter};

/// Determine if a position is "quiet" or not.
/// Currently, we define threats as the ability
/// for the king to escape on the current move.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Threats {
    Quiet,
    Plays(Vec<GameTreeNode>),
}

pub trait SelectionPolicy {
    type TreeNode;
    /// Get the heuristic's evaluation of the position
    /// for the attacking player
    fn eval_attacker(&self, child: &Self::TreeNode) -> i64;
    /// Get the heuristic's evaluation of the position
    /// for the defending player
    fn eval_defender(&self, child: &Self::TreeNode) -> i64;
    /// Given a game node and two indices of it children, figure out which one is better
    /// to explore.
    fn compare_children(
        &self,
        parent: &Self::TreeNode,
        child1: &Self::TreeNode,
        child2: &Self::TreeNode,
    ) -> std::cmp::Ordering;
}

#[derive(Clone, Default)]
pub struct GameTreeNode {
    pub status: Status,
    pub previous_boards: PreviousBoards,
    pub turn: Role,
    pub current_board: Board,
}

impl Debug for GameTreeNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Game")
            .field("status", &self.status)
            .field("turn", &self.turn)
            .field("previous_boards", &self.previous_boards.0.len())
            .field("current_board", &self.current_board.to_string())
            .finish()
    }
}
impl PartialEq for GameTreeNode {
    fn eq(&self, other: &Self) -> bool {
        self.status == other.status
            && self.previous_boards == other.previous_boards
            && self.turn == other.turn
            && self.current_board == other.current_board
    }
}
impl Eq for GameTreeNode{}

impl GameTreeNode {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    fn play(
        &self,
        from: Square,
        to: Square,
        normalized_games: &mut PreviousBoards,
    ) -> Option<Self>
    {
        let play = Play {
            role: self.turn,
            from,
            to,
        };
        let mut game = self.clone();
        if let Ok((_, status)) =
            game.current_board
                .play(&play, &game.status, &mut game.previous_boards)
        {
            let mut normalized = game.clone().current_board;
            normalized.normalize();
            game.status = status;
            game.turn = game.turn.opposite();
            if normalized_games.0.insert(normalized) {
                return Some(game);
            };
        }
        None
    }
    /// Get a vector of child games from this game by checking all
    /// legal moves. We discard children that are symmetrically
    /// equivalent to others.
    pub fn get_children(&self) -> Vec<GameTreeNode> {
        let mut normalized = PreviousBoards::default();
        let mut children = vec![];
        for from in Square::iter() {
            for to in Square::iter() {
                if let Some(node) = self.play(from, to, &mut normalized) {
                    children.push(node);
                }
            }
        }
        children
    }

    /// Get an iterator over the child games from this game by checking all
    /// legal moves. We discard children that are symmetrically
    /// equivalent to others.
    pub fn children(self) -> ChildIterator {
        ChildIterator {
            node: self,
            from: Square::iter(),
            to: Square::iter(),
            normalized: Default::default(),
        }
    }
    pub fn is_terminal(&self) -> bool {
        !matches!(self.status, Status::Ongoing)
    }

    pub fn select_child<S: SelectionPolicy<TreeNode = GameTreeNode>>(&self, policy: &S) -> GameTreeNode {
        let legal_actions = match self.threats() {
            Threats::Quiet => self.get_children(),
            Threats::Plays(threats) => threats,
        };
        if legal_actions.is_empty() {
            unreachable!();
        }

        legal_actions
            .into_iter()
            .max_by(|child1, child2| {
                policy
                    .compare_children(self, child1, child2)
            })
            .unwrap()
    }

    pub fn get_result(&self, for_player: &Role) -> f64 {
        match for_player {
            Role::Attacker => match self.status {
                Status::AttackersWin => 1.0,
                Status::DefendersWin => -1.0,
                Status::Ongoing => unreachable!(),
                Status::Draw => 0.0,
            },
            Role::Defender => match self.status {
                Status::AttackersWin => -1.0,
                Status::DefendersWin => 1.0,
                Status::Ongoing => unreachable!(),
                Status::Draw => 0.0,
            },
        }
    }

    /// Return a list of threats. If there are none, label the position
    /// quiet. This is subjective and will be used to tweak the performance
    /// of the final AI in the endgame.
    pub fn threats(&self) -> Threats {
        if let Role::Defender = self.turn {
            let mut boards = HashSet::with_capacity(4);
            let Some(king) = self.current_board.find_the_king() else {
                return Threats::Quiet;
            };
            let mut threats = Vec::with_capacity(4);
            for corner in EXIT_SQUARES {
                let play = Play {
                    role: Role::Defender,
                    from: king,
                    to: corner,
                };
                let mut game = self.clone();
                if let Ok((_, status)) =
                    game.current_board
                        .play(&play, &game.status, &mut game.previous_boards)
                {
                    game.current_board.normalize();
                    game.status = status;
                    game.turn = game.turn.opposite();
                    if boards.insert(game.current_board.clone()) {
                        threats.push(game)
                    }
                }
            }
            if threats.is_empty() {
                Threats::Quiet
            } else {
                Threats::Plays(threats)
            }
        } else {
            Threats::Quiet
        }
    }
}

/// A iterator over child nodes of a node in the game tree.
/// Only returns normalized boards in an attempt to reduce
/// the branching factor.
pub struct ChildIterator {
    pub node: GameTreeNode,
    pub from: SquareIter,
    pub to: SquareIter,
    pub normalized: PreviousBoards,
}

impl Iterator for ChildIterator {
    type Item = GameTreeNode;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(from) = self.from.next() {
            while let Some(to) = self.to.next() {
                if let Some(node) = self.node
                    .play(from, to, &mut self.normalized)
                {
                    return Some(node);
                }
            }
            self.to = Square::iter();
        }
        None
    }
}



/// An abbreviated view of a game state. Used when game history is
/// not needed to minimize space usage.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct GameSummary {
    pub status: Status,
    pub moves: usize,
    pub turn: Role,
    pub current_board: Board,
}

impl From<&GameTreeNode> for GameSummary {
    fn from(node: &GameTreeNode) -> Self {
        Self {
            status: node.status,
            moves: node.previous_boards.0.len(),
            turn: node.turn,
            current_board: node.current_board.clone(),
        }
    }
}