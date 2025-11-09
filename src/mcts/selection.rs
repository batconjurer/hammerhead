use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::{Device, Tensor};

use crate::game::Status;
use crate::game::space::{Role, Space, Square};
use crate::game_tree::{GameSummary, GameTreeNode, SelectionPolicy};
use crate::mcts::{NNetRole, float_to_scaled_i64, scaled_i64_to_float};

#[derive(Default, Debug)]
pub struct Stats {
    pub visits: AtomicU64,
    pub attacker_rewards: AtomicI64,
    pub defender_rewards: AtomicI64,
}

impl Stats {
    pub fn increment_visits(&self) {
        self.visits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_rewards(&self, player: Role, reward: f64) {
        let reward = float_to_scaled_i64(reward);
        match player {
            Role::Attacker => {
                self.attacker_rewards.fetch_add(reward, Ordering::Relaxed);
            }
            Role::Defender => {
                self.defender_rewards.fetch_add(reward, Ordering::Relaxed);
            }
        }
    }
}

impl TryFrom<&GameSummary> for Tensor {
    type Error = candle_core::Error;

    fn try_from(game: &GameSummary) -> Result<Self, Self::Error> {
        let (attackers, defenders) = Square::iter()
            .map(|sq| match game.current_board.get(&sq) {
                Space::Occupied(Role::Attacker) => (1f64, 0f64),
                Space::Occupied(Role::Defender) => (0.0, 1.0),
                Space::King => (0.0, 2.0),
                Space::Empty => (0.0, 0.0),
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        Tensor::from_vec(
            attackers
                .into_iter()
                .chain(defenders)
                .chain(
                    [if game.turn == Role::Attacker {
                        1f64
                    } else {
                        0f64
                    }; 11 * 11],
                )
                .chain([game.moves as f64; 11 * 11])
                .collect(),
            (4, 11, 11),
            &Device::Cpu,
        )
    }
}

/// A struct holding the current data about how moves are selected.
/// This includes two neural networks, a constant to balance exploration
/// vs. exploitation, and statistics gathered about the result of selections
/// across playouts.
#[derive(Clone)]
pub struct NNSelectionPolicy {
    pub attacker_nn: Option<NNetRole>,
    pub defender_nn: Option<NNetRole>,
    pub exploration_constant: f64,
    pub stats_map: Arc<Mutex<HashMap<GameSummary, Stats>>>,
}

impl Default for NNSelectionPolicy {
    fn default() -> Self {
        Self {
            attacker_nn: None,
            defender_nn: None,
            exploration_constant: 0.2,
            stats_map: Arc::new(Mutex::new(Default::default())),
        }
    }
}

impl NNSelectionPolicy {
    /// The heuristic value from MCTS to be used when a neural network
    /// is not present
    pub fn fallback_eval(&self, child: &GameTreeNode) -> f64 {
        let child_summary = child.into();
        let base = if let Some(stats) = self.stats_map.lock().unwrap().get(&child_summary) {
            scaled_i64_to_float(match child.turn {
                Role::Attacker => stats.attacker_rewards.load(Ordering::Relaxed),
                Role::Defender => stats.defender_rewards.load(Ordering::Relaxed),
            }) / std::cmp::max(stats.visits.load(Ordering::Relaxed), 1) as f64
        } else {
            match child_summary.status {
                Status::AttackersWin => match child.turn {
                    Role::Attacker => 1.0,
                    Role::Defender => -1.0,
                },
                Status::DefendersWin => match child.turn {
                    Role::Attacker => -1.0,
                    Role::Defender => 1.0,
                },
                Status::Draw | Status::Ongoing => 0.0,
            }
        };
        base
    }

    /// Get the number of times this game has been visited
    pub fn get_visits(&self, game: &GameTreeNode) -> u64 {
        let summary = game.into();
        self.stats_map
            .lock()
            .unwrap()
            .get(&summary)
            .map(|stats| stats.visits.load(Ordering::Relaxed))
            .unwrap_or_default()
    }

    /// Update the statistics for a visited node in the tree
    pub fn update_stats(&self, game: &GameTreeNode, attacker_rewards: f64, defender_rewards: f64) {
        let mut stats = self.stats_map.lock().unwrap();
        let summary = game.into();
        match stats.entry(summary) {
            Entry::Occupied(entry) => {
                entry.get().increment_visits();
                entry.get().add_rewards(Role::Attacker, attacker_rewards);
                entry.get().add_rewards(Role::Defender, defender_rewards);
            }
            Entry::Vacant(entry) => {
                entry.insert(Stats {
                    visits: AtomicU64::new(1),
                    attacker_rewards: AtomicI64::new(float_to_scaled_i64(attacker_rewards)),
                    defender_rewards: AtomicI64::new(float_to_scaled_i64(defender_rewards)),
                });
            }
        }
    }

    /// An adjustment added to a positions score to encourage exploration vs. exploitation
    /// This factor should be tightened as models get stronger.
    fn exploration_adjustment(&self, parent: &GameTreeNode, child: &GameTreeNode) -> f64 {
        let child_visits = self.get_visits(child) as f64;
        let parent_visits = std::cmp::max(self.get_visits(parent), 1) as f64;
        self.exploration_constant * (parent_visits.ln() / child_visits).sqrt()
    }
}

impl SelectionPolicy for NNSelectionPolicy {
    type TreeNode = GameTreeNode;

    fn eval_attacker(&self, child: &GameTreeNode) -> i64 {
        let tensor = (&GameSummary::from(child)).try_into().unwrap();
        float_to_scaled_i64(
            self.attacker_nn
                .as_ref()
                .map(|nn| nn.eval(&tensor))
                .unwrap_or_else(|| self.fallback_eval(child)),
        )
    }

    fn eval_defender(&self, child: &GameTreeNode) -> i64 {
        let tensor = (&GameSummary::from(child)).try_into().unwrap();
        float_to_scaled_i64(
            self.defender_nn
                .as_ref()
                .map(|nn| nn.eval(&tensor))
                .unwrap_or_else(|| self.fallback_eval(child)),
        )
    }

    fn compare_children(
        &self,
        parent: &GameTreeNode,
        child1: &GameTreeNode,
        child2: &GameTreeNode,
    ) -> std::cmp::Ordering {
        match parent.turn {
            Role::Defender => {
                let mut eval1 = self.eval_attacker(child1);
                let mut eval2 = self.eval_attacker(child2);
                if NNetRole::is_training(self.attacker_nn.as_ref()) {
                    eval1 += float_to_scaled_i64(self.exploration_adjustment(parent, child1));
                    eval2 += float_to_scaled_i64(self.exploration_adjustment(parent, child2));
                }
                eval1
                    .partial_cmp(&eval2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            Role::Attacker => {
                let mut eval1 = self.eval_defender(child1);
                let mut eval2 = self.eval_defender(child2);
                if NNetRole::is_training(self.defender_nn.as_ref()) {
                    eval1 += float_to_scaled_i64(self.exploration_adjustment(parent, child1));
                    eval2 += float_to_scaled_i64(self.exploration_adjustment(parent, child2));
                }
                eval1
                    .partial_cmp(&eval2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }
    }
}
