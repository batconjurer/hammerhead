use std::sync::{Arc, Mutex};
use crate::game::board::Board;
use crate::game::space::{Role, Square};
use crate::game::{Play, PreviousBoards, Status};
use arboriter_mcts::policy::StandardPolicy;
use arboriter_mcts::policy::selection::{SelectionPolicy, UCB1Policy};
use arboriter_mcts::{Action, BackpropagationPolicy, GameState, MCTSNode, Player};
use crate::nn::TaflNNet;

#[derive(Clone)]
pub struct NNSelectionPolicy {
    attacker_nn: Arc<Mutex<TaflNNet>>,
    defender_nn: Arc<Mutex<TaflNNet>>,
    fallback: UCB1Policy,
}

impl SelectionPolicy<Game> for NNSelectionPolicy {
    fn select_child(&self, node: &MCTSNode<Game>) -> usize {
        self.fallback.select_child(node)
    }

    fn clone_box(&self) -> Box<dyn SelectionPolicy<Game>> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Game {
    pub status: Status,
    pub previous_boards: PreviousBoards,
    pub turn: Role,
    pub current_board: Board,
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
        match self.status {
            Status::Ongoing => false,
            _ => true,
        }
    }

    fn get_result(&self, for_player: &Self::Player) -> f64 {
        match for_player {
            Role::Attacker => match self.status {
                Status::AttackersWin => 1.0,
                Status::DefendersWin => 0.0,
                Status::Ongoing => unreachable!(),
            },
            Role::Defender => match self.status {
                Status::AttackersWin => 0.0,
                Status::DefendersWin => 1.0,
                Status::Ongoing => unreachable!(),
            },
        }
    }

    fn get_current_player(&self) -> Self::Player {
        self.turn
    }
}

#[derive(Clone)]
pub struct NNBackpropogation {
    attacker_nn: (),
    defender_nn: (),
    fallback: StandardPolicy,
}

impl BackpropagationPolicy<Game> for NNBackpropogation {
    fn update_stats(&self, node: &mut MCTSNode<Game>, result: f64) {
        self.fallback.update_stats(node, result);
    }

    fn clone_box(&self) -> Box<dyn BackpropagationPolicy<Game>> {
        Box::new(self.clone())
    }
}
