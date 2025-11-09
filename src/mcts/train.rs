use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::game::PositionsTracker;
use crate::game::space::Role;
use crate::game_tree::{GameSummary, GameTreeNode};
use crate::mcts::selection::{NNSelectionPolicy, Stats};
use crate::mcts::{NNetRole, scaled_i64_to_float};
use candle_core::{Device, Tensor};

pub const ATTACKER_NN_FILE_PREFIX: &str = "hnefatafl_attacker";
pub const DEFENDER_NN_FILE_PREFIX: &str = "hnefatafl_defender";

pub fn train(iterations: usize) {
    // v0 runs
    {
        let defender_nn = NNetRole::training(format!("{}_v0.model", DEFENDER_NN_FILE_PREFIX));
        let stats = Arc::new(Mutex::new(Default::default()));
        let selection_policy = NNSelectionPolicy {
            attacker_nn: None,
            defender_nn: None,
            exploration_constant: 1.414,
            stats_map: stats.clone(),
        };
        let game = GameTreeNode::new(PositionsTracker::Counter(0));
        crate::mcts::mcts(&game, &selection_policy, iterations);
        println!("Finished search");
        let stats = Arc::into_inner(stats).unwrap().into_inner().unwrap();
        backpropagate(defender_nn, &stats);
    }
    {
        let attacker_nn = NNetRole::training(format!("{}_v0.model", ATTACKER_NN_FILE_PREFIX));
        let defender_nn = NNetRole::playing(format!("{}_v0.model", DEFENDER_NN_FILE_PREFIX));
        let stats = Arc::new(Mutex::new(Default::default()));
        let selection_policy = NNSelectionPolicy {
            attacker_nn: Some(attacker_nn.clone()),
            defender_nn: Some(defender_nn.clone()),
            exploration_constant: 1.414,
            stats_map: stats.clone(),
        };
        let game = GameTreeNode::new(PositionsTracker::Counter(0));
        crate::mcts::mcts(&game, &selection_policy, iterations);
        let stats = Arc::into_inner(stats).unwrap().into_inner().unwrap();
        backpropagate(attacker_nn, &stats);
    }
}

fn backpropagate(nn: NNetRole, stats: &HashMap<GameSummary, Stats>) {
    let NNetRole::Training(nn_ptr) = nn else {
        return;
    };
    let mut nn = Arc::into_inner(nn_ptr).unwrap().into_inner().unwrap();
    println!("Training...");
    for (game_pos, stats) in stats {
        for board in game_pos.current_board.symmetries() {
            let game = GameSummary {
                current_board: board,
                ..game_pos.clone()
            };
            let tensor = Tensor::try_from(&game).unwrap();
            let rewards = scaled_i64_to_float(match game.turn {
                Role::Attacker => stats.attacker_rewards.load(Ordering::Relaxed),
                Role::Defender => stats.defender_rewards.load(Ordering::Relaxed),
            });
            // normalize the rewards
            let rewards = rewards / stats.visits.load(Ordering::Relaxed) as f64;
            let rewards = Tensor::new(&[rewards], &Device::Cpu).unwrap();
            nn.train(&tensor, &rewards, 10).unwrap()
        }
    }
}
