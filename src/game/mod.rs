use crate::game::space::{Role, Square};
use board::Board;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

pub mod board;
pub mod space;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PreviousBoards(pub FxHashSet<Board>);

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum Status {
    AttackersWin,
    Draw,
    #[default]
    Ongoing,
    DefendersWin,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Play {
    pub role: Role,
    pub from: Square,
    pub to: Square,
}

impl Play {
    pub fn valid(&self) -> anyhow::Result<()> {
        if [self.from.x, self.from.y].iter().max().unwrap() > &10 {
            return Err(anyhow::Error::msg(
                "The piece to be moved must be on a square on the board.",
            ));
        }
        if [self.to.x, self.to.y].iter().max().unwrap() > &10 {
            return Err(anyhow::Error::msg(
                "The piece must be moved to a square on the board.",
            ));
        }
        let x_diff = self.from.x as i32 - self.to.x as i32;
        let y_diff = self.from.y as i32 - self.to.y as i32;

        if x_diff != 0 && y_diff != 0 {
            return Err(anyhow::Error::msg(
                "play: you can only play in a straight line",
            ));
        }

        if x_diff == 0 && y_diff == 0 {
            return Err(anyhow::Error::msg("play: you have to change location"));
        }

        Ok(())
    }
}
