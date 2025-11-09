use std::io;
use std::io::Write;
use std::process::exit;
use std::str::FromStr;

use crate::game::space::{Role, Square};
use crate::game::{EngineRole, LiveGame, Play, Status};
use clap::{Parser, Subcommand};
use tracing_subscriber::fmt::SubscriberBuilder;

mod alpha_beta;
mod game;
mod game_tree;
mod mcts;
mod nn;

#[derive(Parser)]
#[command(version, about, long_about=None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Make moves on a board in a non-game setting.")]
    Explore,
    #[command(about = "Play against a rudimentary AI")]
    Play { role: Role },
    #[command(about = "Train an AI via self play.")]
    Train {
        #[arg(help = "The number of improved versions to create.")]
        iterations: u64,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum GameCommand {
    Undo,
    Redo,
    Play([Square; 2]),
}

impl FromStr for GameCommand {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "u" | "undo" => Ok(Self::Undo),
            "r" | "redo" => Ok(Self::Redo),
            play => {
                let mut squares = play.split("->");
                let from = Square::from_str(squares.next().ok_or_else(|| {
                    anyhow::Error::msg(format!("Could not parse input '{play}'"))
                })?)?;
                let to = Square::from_str(squares.next().ok_or_else(|| {
                    anyhow::Error::msg(format!("Could not parse input '{play}'"))
                })?)?;
                Ok(Self::Play([from, to]))
            }
        }
    }
}

#[allow(dead_code)]
fn init_logging() {
    SubscriberBuilder::default().with_ansi(true).init();
}

fn main() {
    let cli = Args::parse();
    match cli.command {
        Commands::Explore => explore(None),
        Commands::Train { iterations } => mcts::train(iterations as usize),
        Commands::Play { role } => explore(Some(role)),
    }
    // let mut game = LiveGame::default();
    // game.engine = Some(EngineRole::from(Role::Attacker));
    // game.engine_play();
}

fn user_input() -> GameCommand {
    println!();
    loop {
        print!("Input command: ");
        io::stdout().flush().unwrap();
        let mut buffer = String::new();
        if io::stdin().read_line(&mut buffer).is_err() {
            continue;
        };
        match GameCommand::from_str(buffer.trim()) {
            Ok(command) => return command,
            Err(e) => {
                print!("\x1B[2A\x1B[J");
                io::stdout().flush().unwrap();
                println!("{e}");
            }
        }
        core::hint::spin_loop();
    }
}

fn explore(role: Option<Role>) {
    let mut game = LiveGame {
        engine: role.map(|r| EngineRole::from(r.opposite())),
        ..Default::default()
    };
    loop {
        game.engine_play();
        println!("{}", game);
        match user_input() {
            GameCommand::Undo => game.undo(),
            GameCommand::Redo => game.redo(),
            GameCommand::Play([from, to]) => {
                if let Err(e) = game.play(&Play {
                    role: game.turn,
                    from,
                    to,
                }) {
                    println!("Illegal move: {e}");
                }
            }
        }
        match game.status {
            Status::AttackersWin => {
                println!("Attackers win!");
                exit(0)
            }
            Status::DefendersWin => {
                println!("Defenders win!");
                exit(0)
            }
            _ => {}
        }
    }
}
