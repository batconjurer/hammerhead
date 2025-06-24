use std::sync::atomic::AtomicU64;

use arboriter_mcts::policy::UCB1Policy;
use arboriter_mcts::{GameState, MCTSNode, SelectionPolicy};
use candle_core::Tensor;

use crate::game::space::Role;
use crate::mcts::{Game, NNetRole, Threats};

#[derive(Clone)]
pub struct NNSelectionPolicy {
    pub attacker_nn: Option<NNetRole>,
    pub defender_nn: Option<NNetRole>,
    pub fallback: UCB1Policy,
}

impl NNSelectionPolicy {
    /// Get the neural network's evaluation of the position
    /// for the attacking player
    fn eval_attacker(&self, tensor: &Tensor) -> f64 {
        self.attacker_nn
            .as_ref()
            .map(|nn| nn.eval(tensor))
            .unwrap_or_default()
    }

    /// Get the neural network's evaluation of the position
    /// for the defending player
    fn eval_defender(&self, tensor: &Tensor) -> f64 {
        self.defender_nn
            .as_ref()
            .map(|nn| nn.eval(tensor))
            .unwrap_or_default()
    }

    /// An adjustment added to a positions score to encourage exploration vs. exploitation
    /// This factor should be tightened as models get stronger.
    fn exploration_adjustment(&self, node: &MCTSNode<Game>, child_visits: &AtomicU64) -> f64 {
        use core::sync::atomic::Ordering;
        let child_visits = child_visits.load(Ordering::Relaxed) as f64;
        let total_visits = std::cmp::max(node.visits.load(Ordering::Relaxed), 1) as f64;
        self.fallback.exploration_constant * (total_visits.ln() / child_visits).sqrt()
    }

    /// Given a game node and two indices of it children, figure out which one is better
    /// to explore.
    fn compare_children(
        &self,
        node: &MCTSNode<Game>,
        child1: usize,
        child2: usize,
    ) -> std::cmp::Ordering {
        let MCTSNode {
            state: state1,
            visits: v1,
            ..
        } = &node.children[child1];
        let MCTSNode {
            state: state2,
            visits: v2,
            ..
        } = &node.children[child2];
        match node.player {
            Role::Attacker => {
                let mut eval1 = self.eval_attacker(&state1.try_into().unwrap());
                let mut eval2 = self.eval_attacker(&state1.try_into().unwrap());
                if NNetRole::is_training(self.attacker_nn.as_ref()) {
                    eval1 += self.exploration_adjustment(node, v1);
                    eval2 += self.exploration_adjustment(node, v2);
                }
                eval1
                    .partial_cmp(&eval2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            Role::Defender => {
                let mut eval1 = self.eval_defender(&state2.try_into().unwrap());
                let mut eval2 = self.eval_defender(&state2.try_into().unwrap());
                if NNetRole::is_training(self.defender_nn.as_ref()) {
                    eval1 += self.exploration_adjustment(node, v1);
                    eval2 += self.exploration_adjustment(node, v2);
                }
                eval1
                    .partial_cmp(&eval2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }
    }
}

impl SelectionPolicy<Game> for NNSelectionPolicy {
    fn select_child(&self, node: &MCTSNode<Game>) -> usize {
        if let Threats::Plays(plays) = node.state.threats() {
            let state = node.state.apply_action(&plays[0]);
            return node
                .children
                .iter()
                .enumerate()
                .find(|(_, p)| p.state == state)
                .unwrap()
                .0;
        }
        match node.player {
            Role::Attacker if self.attacker_nn.is_some() => (0..node.children.len())
                .max_by(|ix1, ix2| self.compare_children(node, *ix1, *ix2))
                .unwrap_or_default(),
            Role::Defender if self.defender_nn.is_some() => (0..node.children.len())
                .max_by(|ix1, ix2| self.compare_children(node, *ix1, *ix2))
                .unwrap_or_default(),
            _ => self.fallback.select_child(node),
        }
    }

    fn clone_box(&self) -> Box<dyn SelectionPolicy<Game>> {
        Box::new(self.clone())
    }
}
