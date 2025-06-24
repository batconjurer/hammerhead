use crate::game::space::Role;
use crate::mcts::{Game, NNetRole};
use arboriter_mcts::policy::StandardPolicy;
use arboriter_mcts::{BackpropagationPolicy, MCTSNode};
use std::sync::atomic::Ordering;

#[derive(Clone)]
pub struct NNBackpropogation {
    pub attacker_nn: Option<NNetRole>,
    pub defender_nn: Option<NNetRole>,
    pub fallback: StandardPolicy,
}

impl BackpropagationPolicy<Game> for NNBackpropogation {
    fn update_stats(&self, node: &mut MCTSNode<Game>, result: f64) {
        self.fallback.update_stats(node, result);
        let reward = node.total_reward.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        match node.player {
            Role::Attacker if self.attacker_nn.is_some() => self
                .attacker_nn
                .as_ref()
                .unwrap()
                .train(&node.state, reward),
            Role::Defender if self.defender_nn.is_some() => self
                .attacker_nn
                .as_ref()
                .unwrap()
                .train(&node.state, reward),
            _ => {}
        }
    }

    fn clone_box(&self) -> Box<dyn BackpropagationPolicy<Game>> {
        Box::new(self.clone())
    }
}
