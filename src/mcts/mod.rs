mod selection;
mod train;

use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::{Module, Tensor};
//use rayon::prelude::*;
pub use train::train;

use crate::game::board::Board;
use crate::game::space::{EXIT_SQUARES, Role, Square};
use crate::game::{Play, PreviousBoards, Status};
use crate::mcts::selection::{NNSelectionPolicy};
use crate::nn::TaflNNet;

/// Internal representation of a fixed-point value for rewards
/// This allows atomic operations on floating point rewards
const REWARD_SCALE: f64 = 1_000_000.0;

/// Safely convert a floating point reward to a scaled integer
fn float_to_scaled_u64(value: f64) -> u64 {
    ((value * REWARD_SCALE).max(0.0) as u64).min(u64::MAX / 2)
}

/// Safely convert a scaled integer back to a floating point reward
fn scaled_u64_to_float(value: u64) -> f64 {
    value as f64 / REWARD_SCALE
}

/// Run Monte Carlo tree search on the given starting position for the given
/// number of iterations. Return the selection policy afterwards.
pub fn mcts(root: Game, iterations: usize) {
    println!("Playing {iterations} games");
    for _ in 0..iterations {
        root.simulate_random_playout();
    }
}

/// Determine if a position is "quiet" or not.
/// Currently, we define threats as the ability
/// for the king to escape on the current move.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Threats {
    Quiet,
    Plays(Vec<Game>),
}


#[derive(Clone, Default)]
pub struct Game {
    pub status: Status,
    pub previous_boards: PreviousBoards,
    pub turn: Role,
    pub current_board: Board,
    pub selection: NNSelectionPolicy,
}

impl Debug for Game {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Game")
            .field("status", &self.status)
            .field("turn", &self.turn)
            .field("previous_boards", &self.previous_boards.0.len())
            .field("current_board", &self.current_board.to_string())
            .finish()
    }
}
impl PartialEq for Game {
    fn eq(&self, other: &Self) -> bool {
        self.status == other.status
            && self.previous_boards == other.previous_boards
            && self.turn == other.turn
            && self.current_board == other.current_board
    }
}
impl Eq for Game {}


impl Game {

    pub fn new(selection_policy: NNSelectionPolicy) -> Self {
        Self {
            selection: selection_policy,
            ..Default::default()
        }
    }

    /// Get a set of child games from this game by checking all
    /// legal moves. We discard children that are symmetrically
    /// equivalent to others.
    pub fn get_children(&self) -> Vec<Game> {
        let mut normalized_children = HashSet::new();
        let mut children = vec![];
        for from in Square::iter() {
            for to in Square::iter() {
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
                    if normalized_children.insert(normalized) {
                        children.push(game);
                    };
                }
            }
        }
        children
    }

    fn is_terminal(&self) -> bool {
        !matches!(self.status, Status::Ongoing)
    }

    fn get_result(&self, for_player: &Role) -> f64 {
        match for_player {
            Role::Attacker => match self.status {
                Status::AttackersWin => 1.0,
                Status::DefendersWin => 0.0,
                Status::Ongoing => unreachable!(),
                Status::Draw => 0.0,
            },
            Role::Defender => match self.status {
                Status::AttackersWin => 0.0,
                Status::DefendersWin => 1.0,
                Status::Ongoing => unreachable!(),
                Status::Draw => 0.0,
            },
        }
    }

    fn simulate_random_playout(&self) -> f64 {
        let mut current_state = self.clone();
        let for_player = self.turn;
        let mut path = Vec::from([current_state.clone()]);
        while !current_state.is_terminal() {
            let legal_actions = match current_state.threats() {
                Threats::Quiet => current_state.get_children(),
                Threats::Plays(threats) => threats,
            };
            if legal_actions.is_empty() {
                unreachable!();
            }

            current_state = legal_actions
                .into_iter()
                .max_by(|child1, child2| {
                    self.selection
                        .compare_children(&current_state, child1, child2)
                })
                .unwrap();
            path.push(current_state.clone());
        }
        if current_state.status == Status::AttackersWin {
            println!("Attacker victory");
        }
        let attacker_rewards = current_state.get_result(&Role::Attacker);
        let defender_rewards = current_state.get_result(&Role::Defender);
        for game in path {
            self.selection
                .update_stats(&game, attacker_rewards, defender_rewards);
        }
        match for_player {
            Role::Attacker => attacker_rewards,
            Role::Defender => defender_rewards,
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
    use super::{Game, Threats};
    use crate::game::Play;
    use crate::game::board::Board;
    use crate::game::space::{Role, Square};

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
        let mut game = Game {
            status: Default::default(),
            previous_boards: Default::default(),
            turn: Role::Attacker,
            current_board: Board::try_from(board).expect("Test failed"),
            selection: Default::default(),
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
        let expected = expected_plays.iter().map(|play| {
            let mut g = game.clone();
            g.current_board.play(play, &g.status, &mut g.previous_boards).expect("Test failed");
            g.current_board.normalize();
            g.current_board
        }).collect::<Vec<_>>();
        let threats = match game.threats() {
            Threats::Quiet => panic!("Test failed"),
            Threats::Plays(games) => games.into_iter()
                .map(|g| g.current_board)
                .collect::<Vec<_>>()
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
        let game = Game {
            status: Default::default(),
            previous_boards: Default::default(),
            turn: Role::Defender,
            current_board: Board::try_from(board).expect("Test failed"),
            selection: Default::default(),
        };
        let expected_plays = vec![Play {
            role: Role::Defender,
            from: Square { x: 0, y: 8 },
            to: Square { x: 0, y: 10 },
        }];
        let expected = expected_plays.iter().map(|play| {
            let mut g = game.clone();
            g.current_board.play(play, &g.status, &mut g.previous_boards).expect("Test failed");
            g.current_board.normalize();
            g.current_board
        }).collect::<Vec<_>>();
        let threats = match game.threats() {
            Threats::Quiet => panic!("Test failed"),
            Threats::Plays(games) => games.into_iter()
                .map(|g| g.current_board)
                .collect::<Vec<_>>()
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
        let game = Game {
            status: Default::default(),
            previous_boards: Default::default(),
            turn: Role::Defender,
            current_board: Board::try_from(board).expect("Test failed"),
            selection: Default::default(),
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
        let game = Game {
            status: Default::default(),
            previous_boards: Default::default(),
            turn: Role::Defender,
            current_board: Board::try_from(board).expect("Test failed"),
            selection: Default::default(),
        };
        assert_eq!(Threats::Quiet, game.threats());
    }

}
