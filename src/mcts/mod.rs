mod backpropogation;
mod selection;
mod train;

use std::path::Path;
use std::sync::{Arc, Mutex};

use arboriter_mcts::{Action, GameState, Player};
use candle_core::{Device, Module, Tensor};
pub use train::train;

use crate::game::board::Board;
use crate::game::space::{EXIT_SQUARES, Role, Space, Square};
use crate::game::{Play, PreviousBoards, Status};
use crate::nn::TaflNNet;

/// Determine if a position is "quiet" or not.
/// Currently, we define threats as the ability
/// for the king to escape on the current move.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Threats {
    Quiet,
    Plays(Vec<Play>),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Game {
    pub status: Status,
    pub previous_boards: PreviousBoards,
    pub turn: Role,
    pub current_board: Board,
}

impl Game {
    /// Return a list of threats. If there are none, label the position
    /// quiet. This is subjective and will be used to tweak the performance
    /// of the final AI in the endgame.
    pub fn threats(&self) -> Threats {
        if let Role::Defender = self.turn {
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
                if self
                    .current_board
                    .play_internal(&play, &Status::Ongoing, &self.previous_boards)
                    .is_ok()
                {
                    threats.push(play);
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

impl TryFrom<&Game> for Tensor {
    type Error = candle_core::Error;

    fn try_from(game: &Game) -> Result<Self, Self::Error> {
        let (attackers, defenders) = Square::iter()
            .map(|sq| match game.current_board.get(&sq) {
                Space::Occupied(Role::Attacker) => (1u8, 0u8),
                Space::Occupied(Role::Defender) => (0, 1),
                Space::King => (0, 2),
                Space::Empty => (0, 0),
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        Tensor::from_vec(
            attackers
                .into_iter()
                .chain(defenders)
                .chain(
                    [if game.turn == Role::Attacker {
                        1u8
                    } else {
                        0u8
                    }; 11 * 11],
                )
                .chain([game.previous_boards.0.len() as u8; 11 * 11])
                .collect(),
            (4, 11, 11),
            &Device::Cpu,
        )
    }
}

impl Action for Play {
    fn id(&self) -> usize {
        let mut bytes = [0u8; 8];
        bytes[0] = self.from.x as u8;
        bytes[1] = self.from.y as u8;
        bytes[2] = self.to.x as u8;
        bytes[3] = self.to.y as u8;
        bytes[4] = match self.role {
            Role::Attacker => 0,
            Role::Defender => 1,
        };
        usize::from_le_bytes(bytes)
    }
}

impl Player for Role {}

impl GameState for Game {
    type Action = Play;
    type Player = Role;

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        let mut legal_actions = vec![];
        for from in Square::iter() {
            for to in Square::iter() {
                let play = Play {
                    role: self.turn,
                    from,
                    to,
                };
                if self
                    .current_board
                    .play_internal(&play, &self.status, &self.previous_boards)
                    .is_ok()
                {
                    legal_actions.push(play);
                }
            }
        }
        legal_actions
    }

    fn apply_action(&self, action: &Self::Action) -> Self {
        let mut game = self.clone();
        let (_, status) = game
            .current_board
            .play(action, &self.status, &mut game.previous_boards)
            .expect("The validity of this action should have already been checked");
        game.status = status;
        game.turn = game.turn.opposite();
        game
    }

    fn is_terminal(&self) -> bool {
        !matches!(self.status, Status::Ongoing)
    }

    fn get_result(&self, for_player: &Self::Player) -> f64 {
        match for_player {
            Role::Attacker => match self.status {
                Status::AttackersWin => 1.0,
                Status::DefendersWin => 0.0,
                Status::Ongoing => unreachable!(),
                Status::Draw => 0.5,
            },
            Role::Defender => match self.status {
                Status::AttackersWin => 0.0,
                Status::DefendersWin => 1.0,
                Status::Ongoing => unreachable!(),
                Status::Draw => 0.5,
            },
        }
    }

    fn get_current_player(&self) -> Self::Player {
        self.turn
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
            .to_scalar::<f64>()
            .unwrap()
    }

    /// A helper function to help policies determine if a
    /// given neural network is currently being trained
    fn is_training(nn: Option<&Self>) -> bool {
        matches!(nn, Some(NNetRole::Training(_)))
    }

    /// Train the neural network to learn the evaluation of the given
    /// board position
    fn train(&self, game: &Game, result: f64) {
        let tensor = Tensor::try_from(game).unwrap();
        let result = Tensor::new(&[result], &Device::Cpu).unwrap();
        let Self::Training(nn) = self else { return };
        let mut guard = nn.lock().unwrap();
        // TODO! Should we train on board symmetries here as well?
        guard.train(&tensor, &result, 500).unwrap();
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
        };

        assert_eq!(Threats::Quiet, game.threats());
        game.turn = Role::Defender;
        let expected = vec![
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
        assert_eq!(Threats::Plays(expected), game.threats());
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
        };
        let expected = vec![Play {
            role: Role::Defender,
            from: Square { x: 0, y: 8 },
            to: Square { x: 0, y: 10 },
        }];
        assert_eq!(Threats::Plays(expected), game.threats());
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
        };
        assert_eq!(Threats::Quiet, game.threats());
    }
}
