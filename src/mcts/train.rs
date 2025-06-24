use arboriter_mcts::config::BestChildCriteria;
use arboriter_mcts::policy::{StandardPolicy, UCB1Policy};
use arboriter_mcts::{MCTS, MCTSConfig};

use crate::mcts::backpropogation::NNBackpropogation;
use crate::mcts::selection::NNSelectionPolicy;
use crate::mcts::{Game, NNetRole};

pub const ATTACKER_NN_FILE_PREFIX: &str = "hnefatafl_attacker";
pub const DEFENDER_NN_FILE_PREFIX: &str = "hnefatafl_defender";

pub fn train(_iterations: u64) {
    let config = MCTSConfig::default()
        .with_best_child_criteria(BestChildCriteria::MostVisits)
        .with_exploration_constant(0.2)
        .with_max_iterations(800)
        .with_node_pool_config(1000, 256)
        .with_transpositions(true);
    // v0 runs
    {
        let defender_nn = NNetRole::training(format!("{}_v0.model", DEFENDER_NN_FILE_PREFIX));
        let selection_policy = NNSelectionPolicy {
            attacker_nn: None,
            defender_nn: Some(defender_nn.clone()),
            fallback: UCB1Policy::new(0.2),
        };
        let backprop_policy = NNBackpropogation {
            attacker_nn: None,
            defender_nn: Some(defender_nn),
            fallback: StandardPolicy::new(),
        };
        let mut mcts = MCTS::new(Game::default(), config.clone())
            .with_selection_policy(selection_policy)
            .with_backpropagation_policy(backprop_policy);
        _ = mcts.search().unwrap();
    }
    {
        let attacker_nn = NNetRole::training(format!("{}_v0.model", ATTACKER_NN_FILE_PREFIX));
        let defender_nn = NNetRole::playing(format!("{}_v0.model", DEFENDER_NN_FILE_PREFIX));
        let selection_policy = NNSelectionPolicy {
            attacker_nn: Some(attacker_nn.clone()),
            defender_nn: Some(defender_nn.clone()),
            fallback: UCB1Policy::new(0.2),
        };
        let backprop_policy = NNBackpropogation {
            attacker_nn: Some(attacker_nn),
            defender_nn: Some(defender_nn),
            fallback: StandardPolicy::new(),
        };
        let mut mcts = MCTS::new(Game::default(), config)
            .with_selection_policy(selection_policy)
            .with_backpropagation_policy(backprop_policy);
        _ = mcts.search().unwrap();
    }
}
