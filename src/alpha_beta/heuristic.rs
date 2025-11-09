use once_cell::sync::Lazy;
use std::cmp::Ordering;
use std::sync::Mutex;

use crate::game::board::Board;
use crate::game::heuristics::{escape_routes, fewest_turns_to_escape};
use crate::game::space::{Role, Space, Square};
use crate::game::{NormalizedBoardMap, Status};
use crate::game_tree::{GameTreeNode, SelectionPolicy};
use crate::mcts::{float_to_scaled_i64, scaled_i64_to_float};

/// When the king has no path to any square, an evaluation
/// of that portion of the score.
const UNREACHABLE_ESCAPE_SCORE: u8 = 8;

/// A global table of the heuristic evaluations of board positions from the attacker's standpoint
static BOARD_EVALUATIONS: Lazy<Mutex<NormalizedBoardMap<i64>>> =
    Lazy::new(|| Mutex::new(NormalizedBoardMap::default()));

/// A heuristic evaluation of a game state. It takes into account
/// the following:
///  * If the King can escape
///  * The distance of the king to the nearest escape square
///  * The number of squares needed to be occupied by attackers
///    to block the king from all escapes
///  * The material difference
pub fn heuristic(game: &GameTreeNode) -> i64 {
    match game.status {
        Status::AttackersWin => {
            return float_to_scaled_i64(match game.turn {
                Role::Attacker => 10000.0,
                Role::Defender => -10000.0,
            });
        }
        Status::DefendersWin => {
            return float_to_scaled_i64(match game.turn {
                Role::Attacker => -10000.0,
                Role::Defender => 10000.0,
            });
        }
        Status::Draw => return 0,
        Status::Ongoing => {
            if let Some(val) = BOARD_EVALUATIONS.lock().unwrap().get(&game.current_board) {
                return match game.turn {
                    Role::Attacker => *val,
                    Role::Defender => -*val,
                };
            }
        }
    }

    // a number between 0 and 8
    let escapes = escape_routes(&game.current_board) as i64;
    let escape_dist =
        fewest_turns_to_escape(&game.current_board).unwrap_or(UNREACHABLE_ESCAPE_SCORE) as i64;
    // attackers want to maximize this metric
    let piece_diff =
        (game.current_board.attackers() as i64 - game.current_board.defenders() as i64) - 11;
    let attacker_score = scaled_i64_to_float(piece_diff + escape_dist - escapes)
        + attacker_corner_penalties(&game.current_board);
    BOARD_EVALUATIONS
        .lock()
        .unwrap()
        .insert(&game.current_board, float_to_scaled_i64(attacker_score));
    float_to_scaled_i64(match game.turn {
        Role::Attacker => attacker_score,
        Role::Defender => -attacker_score,
    })
}

/// For each attacker next to a corner which is vulnerable
/// to capture, add a penalty.
fn attacker_corner_penalties(board: &Board) -> f64 {
    const PENALTY_AMOUNT: f64 = 0.5;
    let mut penalty = 0f64;
    if let Space::Occupied(Role::Attacker) = board.get(&Square { x: 1, y: 0 }) {
        if !board.is_occupied(&Square { x: 2, y: 0 }) {
            penalty -= PENALTY_AMOUNT;
        }
    }
    if let Space::Occupied(Role::Attacker) = board.get(&Square { x: 0, y: 1 }) {
        if !board.is_occupied(&Square { x: 0, y: 2 }) {
            penalty -= PENALTY_AMOUNT;
        }
    }
    if let Space::Occupied(Role::Attacker) = board.get(&Square { x: 9, y: 0 }) {
        if !board.is_occupied(&Square { x: 8, y: 0 }) {
            penalty -= PENALTY_AMOUNT;
        }
    }
    if let Space::Occupied(Role::Attacker) = board.get(&Square { x: 10, y: 1 }) {
        if !board.is_occupied(&Square { x: 10, y: 2 }) {
            penalty -= PENALTY_AMOUNT;
        }
    }
    if let Space::Occupied(Role::Attacker) = board.get(&Square { x: 1, y: 10 }) {
        if !board.is_occupied(&Square { x: 2, y: 10 }) {
            penalty -= PENALTY_AMOUNT;
        }
    }
    if let Space::Occupied(Role::Attacker) = board.get(&Square { x: 0, y: 9 }) {
        if !board.is_occupied(&Square { x: 0, y: 8 }) {
            penalty -= PENALTY_AMOUNT;
        }
    }
    if let Space::Occupied(Role::Attacker) = board.get(&Square { x: 9, y: 10 }) {
        if !board.is_occupied(&Square { x: 8, y: 10 }) {
            penalty -= PENALTY_AMOUNT;
        }
    }
    if let Space::Occupied(Role::Attacker) = board.get(&Square { x: 10, y: 9 }) {
        if !board.is_occupied(&Square { x: 10, y: 8 }) {
            penalty -= PENALTY_AMOUNT;
        }
    }
    penalty
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct HeuristicPolicy;

impl SelectionPolicy for HeuristicPolicy {
    type TreeNode = GameTreeNode;

    fn eval_attacker(&self, child: &Self::TreeNode) -> i64 {
        heuristic(child)
    }

    fn eval_defender(&self, child: &Self::TreeNode) -> i64 {
        heuristic(child)
    }

    fn compare_children(
        &self,
        parent: &Self::TreeNode,
        child1: &Self::TreeNode,
        child2: &Self::TreeNode,
    ) -> Ordering {
        match parent.turn {
            Role::Attacker => self.eval_defender(child2).cmp(&self.eval_defender(child1)),
            Role::Defender => self.eval_attacker(child1).cmp(&self.eval_attacker(child2)),
        }
    }
}

#[cfg(test)]
mod test_heuristic {
    use super::*;
    use crate::alpha_beta::alphabeta_inner;
    use crate::game::{EngineRole, LiveGame, Play};
    use crate::game_tree::GameSummary;
    use rustc_hash::FxHashMap;
    use std::str::FromStr;

    #[test]
    fn test_threatening_position() {
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

        let game = LiveGame {
            current_board: board,
            engine: Some(EngineRole::from(Role::Attacker)),
            ..Default::default()
        };

        let mut alphas: FxHashMap<GameSummary, i64> = FxHashMap::default();
        let mut betas: FxHashMap<GameSummary, i64> = FxHashMap::default();
        let mut non_block = game.clone();
        non_block
            .play(&Play {
                role: Role::Attacker,
                from: Square::from_str("f2").unwrap(),
                to: Square::from_str("g2").unwrap(),
            })
            .expect("Test failed");

        let root = GameTreeNode::from(&mut non_block);
        let res = alphabeta_inner::<GameSummary, _, _>(
            &root,
            &HeuristicPolicy,
            &mut alphas,
            &mut betas,
            3,
        );
        assert_eq!(res, float_to_scaled_i64(-10000.0));

        let mut alphas: FxHashMap<GameSummary, i64> = FxHashMap::default();
        let mut betas: FxHashMap<GameSummary, i64> = FxHashMap::default();
        let mut non_block = game.clone();
        non_block
            .play(&Play {
                role: Role::Attacker,
                from: Square::from_str("f2").unwrap(),
                to: Square::from_str("k2").unwrap(),
            })
            .expect("Test failed");

        let root = GameTreeNode::from(&mut non_block);
        let best_res = alphabeta_inner::<GameSummary, _, _>(
            &root,
            &HeuristicPolicy,
            &mut alphas,
            &mut betas,
            3,
        );
        assert!(best_res > float_to_scaled_i64(-10000.0));
    }
}
