mod selection;
mod train;

use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::{Module, Tensor};
//use rayon::prelude::*;
pub use train::train;

use crate::game::Status;
use crate::game::space::Role;
use crate::game_tree::GameTreeNode;
use crate::mcts::selection::NNSelectionPolicy;
use crate::nn::TaflNNet;

/// Internal representation of a fixed-point value for rewards
/// This allows atomic operations on floating point rewards
const REWARD_SCALE: f64 = 1_000_000.0;
const REWARD_MIN: f64 = -9223372036854775000.0;
/// Safely convert a floating point reward to a scaled integer
pub fn float_to_scaled_i64(value: f64) -> i64 {
    (value * REWARD_SCALE).max(REWARD_MIN).trunc() as i64
}

/// Safely convert a scaled integer back to a floating point reward
pub fn scaled_i64_to_float(value: i64) -> f64 {
    (value as f64) / REWARD_SCALE
}
/// Run Monte Carlo tree search on the given starting position for the given
/// number of iterations. Return the selection policy afterwards.
pub fn mcts(root: &GameTreeNode, policy: &NNSelectionPolicy, iterations: usize) {
    println!("Playing {iterations} games");
    for _ in 0..iterations {
        simulate_random_playout(root, policy);
    }
}
pub fn simulate_random_playout(node: &GameTreeNode, policy: &NNSelectionPolicy) -> f64 {
    let mut current_state = node.clone();
    let for_player = node.turn;
    let mut path = Vec::from([current_state.clone()]);
    while !current_state.is_terminal() {
        current_state = current_state.select_child(policy);
        path.push(current_state.clone());
    }
    if current_state.status == Status::AttackersWin {
        println!("Attacker victory");
    }
    let attacker_rewards = current_state.get_result(&Role::Attacker);
    let defender_rewards = current_state.get_result(&Role::Defender);
    for game in path {
        policy.update_stats(&game, attacker_rewards, defender_rewards);
    }
    match for_player {
        Role::Attacker => attacker_rewards,
        Role::Defender => defender_rewards,
    }
}

/// An enum indicating whether a [`TaflNNet`] is being
/// trained or simply being used to play.
#[derive(Clone)]
pub enum NNetRole {
    Training(Arc<Mutex<TaflNNet>>),
    Playing(Arc<Mutex<TaflNNet>>),
}

impl NNetRole {
    /// Open training neural network
    pub fn training(p: impl AsRef<Path>) -> Self {
        NNetRole::Training(Arc::new(Mutex::new(TaflNNet::new(p))))
    }

    /// Open playing neural network
    pub fn playing(p: impl AsRef<Path>) -> Self {
        NNetRole::Playing(Arc::new(Mutex::new(TaflNNet::new(p))))
    }

    /// Get the inner pointer
    fn inner(&self) -> &Arc<Mutex<TaflNNet>> {
        match self {
            NNetRole::Training(nn) => nn,
            NNetRole::Playing(nn) => nn,
        }
    }

    /// Evaluate the inner [`TaflNNet`] on the given tensor and
    /// cast it to a float
    fn eval(&self, tensor: &Tensor) -> f64 {
        self.inner()
            .lock()
            .unwrap()
            .forward(tensor)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
    }

    /// A helper function to help policies determine if a
    /// given neural network is currently being trained
    fn is_training(nn: Option<&Self>) -> bool {
        matches!(nn, Some(NNetRole::Training(_)))
    }
}

#[cfg(test)]
mod tests {

    use crate::game::board::Board;
    use crate::game::space::{Role, Square};
    use crate::game::{Play, PositionsTracker};
    use crate::game_tree::{GameTreeNode, Threats};

    #[test]
    fn test_threats() {
        let board = [
            "...........",
            "...........",
            ".X.........",
            ".X.........",
            ".X.........",
            ".X.........",
            "...........",
            ".X.........",
            "KX.........",
            ".X.........",
            "...........",
        ];
        let mut game = GameTreeNode {
            status: Default::default(),
            previous_boards: PositionsTracker::Counter(0),
            turn: Role::Attacker,
            current_board: Board::try_from(board).expect("Test failed"),
        };

        assert_eq!(Threats::Quiet, game.threats());
        game.turn = Role::Defender;
        let expected_plays = vec![
            Play {
                role: Role::Defender,
                from: Square { x: 0, y: 8 },
                to: Square { x: 0, y: 0 },
            },
            Play {
                role: Role::Defender,
                from: Square { x: 0, y: 8 },
                to: Square { x: 0, y: 10 },
            },
        ];
        let expected = expected_plays
            .iter()
            .map(|play| {
                let mut g = game.clone();
                g.current_board
                    .play(play, &g.status, &mut g.previous_boards)
                    .expect("Test failed");
                g.current_board.normalize();
                g.current_board
            })
            .collect::<Vec<_>>();
        let threats = match game.threats() {
            Threats::Quiet => panic!("Test failed"),
            Threats::Plays(games) => games
                .into_iter()
                .map(|g| g.current_board)
                .collect::<Vec<_>>(),
        };

        assert_eq!(threats, expected);
        let board = [
            "...........",
            "...........",
            ".X.........",
            ".X.........",
            ".X.........",
            ".X.........",
            "...........",
            "OX.........",
            "KX.........",
            ".X.........",
            "...........",
        ];
        let game = GameTreeNode {
            status: Default::default(),
            previous_boards: PositionsTracker::Counter(0),
            turn: Role::Defender,
            current_board: Board::try_from(board).expect("Test failed"),
        };
        let expected_plays = vec![Play {
            role: Role::Defender,
            from: Square { x: 0, y: 8 },
            to: Square { x: 0, y: 10 },
        }];
        let expected = expected_plays
            .iter()
            .map(|play| {
                let mut g = game.clone();
                g.current_board
                    .play(play, &g.status, &mut g.previous_boards)
                    .expect("Test failed");
                g.current_board.normalize();
                g.current_board
            })
            .collect::<Vec<_>>();
        let threats = match game.threats() {
            Threats::Quiet => panic!("Test failed"),
            Threats::Plays(games) => games
                .into_iter()
                .map(|g| g.current_board)
                .collect::<Vec<_>>(),
        };
        assert_eq!(threats, expected);
        let board = [
            "...........",
            "...........",
            ".X.........",
            ".X.........",
            ".X.........",
            ".X.........",
            "...........",
            "OX.........",
            "KX.........",
            "OX.........",
            "...........",
        ];
        let game = GameTreeNode {
            status: Default::default(),
            previous_boards: PositionsTracker::Counter(0),
            turn: Role::Defender,
            current_board: Board::try_from(board).expect("Test failed"),
        };
        assert_eq!(Threats::Quiet, game.threats());
        let board = [
            "...........",
            "...........",
            ".X.........",
            ".X.........",
            ".X.........",
            ".X.........",
            "...........",
            "O..........",
            "....K......",
            "...........",
            "...........",
        ];
        let game = GameTreeNode {
            status: Default::default(),
            previous_boards: PositionsTracker::Counter(0),
            turn: Role::Defender,
            current_board: Board::try_from(board).expect("Test failed"),
        };
        assert_eq!(Threats::Quiet, game.threats());
    }
}
