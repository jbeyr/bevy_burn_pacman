//! Pac-Man Lite — Bevy 0.19 playable game + headless DQN training (Burn 0.21).
//!
//! Default invocation launches the interactive 60 FPS game. `--headless` runs
//! fast-forward training episodes against the scripted ghosts instead.

#![allow(dead_code)]
#![allow(clippy::needless_pass_by_value)]

mod agent;
mod events;
mod ghosts;
mod headless_env;
mod maze;
mod pacman;
mod ui;

use bevy::prelude::*;
use clap::Parser;

/// Top-level application states.
#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum GameState {
    /// Active gameplay: maze, player, and ghosts are live.
    #[default]
    Playing,
    /// All lives exhausted; waiting for restart input.
    GameOver,
    /// Every pellet eaten; level won.
    Win,
}

/// Session-wide gameplay counters reset on level entry.
#[derive(Debug, Resource)]
struct GameSession {
    pellets_at_start: usize,
}

impl GameSession {
    fn new(count: usize) -> Self {
        Self {
            pellets_at_start: count,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Pac-Man Lite - Bevy 0.19 + Burn 0.21")]
pub struct CliArgs {
    /// Run headless training benchmark without spawning a graphical window
    #[arg(long)]
    pub headless: bool,

    /// Number of training episodes to simulate in headless mode
    #[arg(long, default_value_t = 500)]
    pub episodes: usize,

    /// Path to save the trained model checkpoint (.mpk)
    #[arg(long, default_value = "pacman_policy.mpk")]
    pub save: String,

    /// Checkpoint for the GUI agent to load (GUI mode only)
    #[arg(long, default_value = "pacman_policy.mpk")]
    pub model: String,
}


fn main() {
    let args = CliArgs::parse();

    if args.headless {
        run_headless_training(&args);
        return;
    }

    println!("Launching Pac-Man Lite GUI (Bevy 0.19)...");
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Pac-Man Lite - Bevy 0.19".to_string(),
                resolution: bevy::window::WindowResolution::new(1024, 768),
                resizable: true,
                ..default()
            }),
            ..default()
        }))
        .init_state::<GameState>()
        .init_resource::<maze::MazeGrid>()
        .insert_resource(maze::PelletMap::default())
        .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.05)))
        .add_systems(Startup, spawn_camera)
        .add_systems(Startup, (maze::spawn_maze, pacman::spawn_player, ghosts::spawn_ghosts))
        .add_systems(
            Update,
            (
                pacman::handle_input,
                pacman::player_movement,
                ghosts::ghost_movement,
                pacman::check_ghost_collision,
                ghosts::update_mode_timers,
            )
                .run_if(in_state(GameState::Playing)),
        )
        .add_plugins(ui::UiPlugin)
        .insert_resource(GameSession::new(0))
        .add_observer(ghosts::frighten_all)
        .add_systems(
            Update,
            (
                check_level_cleared,
                handle_life_loss.run_if(in_state(GameState::Playing)),
            ),
        )
        .run();
}

/// Spawns the world camera centered on the maze.
fn spawn_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// Detects full pellet consumption and transitions to the win state.
fn check_level_cleared(
    mut next_state: ResMut<NextState<GameState>>,
    session: Option<Res<GameSession>>,
    pellets: Res<maze::PelletMap>,
) {
    let Some(session) = session else {
        return;
    };
    if !pellets.is_changed() {
        return;
    }
    if pellets.remaining() == 0 && session.pellets_at_start > 0 {
        next_state.set(GameState::Win);
    }
}

/// Watches life loss and ends the game when none remain.
fn handle_life_loss(lives: Res<ui::Lives>, mut next_state: ResMut<NextState<GameState>>) {
    if lives.is_changed() && lives.0 == 0 {
        next_state.set(GameState::GameOver);
    }
}

/// High-throughput headless Double DQN training against scripted ghosts.
fn run_headless_training(args: &CliArgs) {
    use agent::{DoubleDqnAgent, ReplayBuffer, Transition};
    use burn::backend::Autodiff;
    use burn::backend::ndarray::NdArray;
    use headless_env::{OBS_DIM, PacEnv};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::time::Instant;

    type AutodiffNdArray = Autodiff<NdArray<f32>>;

    println!(
        "Pac-Man Lite training: {} episodes | horizon {} steps | obs dim {OBS_DIM} | actions 4",
        args.episodes, 600
    );

    let start = Instant::now();
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let mut agent =
        DoubleDqnAgent::<AutodiffNdArray>::new(device, 0.99);
    let mut buffer = ReplayBuffer::new(50_000);
    let mut rng = StdRng::seed_from_u64(0x00C0_FFEE);
    let mut env = PacEnv::new();

    let batch_size = 64usize;
    let lr = 5.0e-4_f64;
    let gamma_decay = 0.995_f32;
    let mut epsilon = 1.0_f32;
    let min_epsilon = 0.05_f32;

    let mut obs = [0.0f32; OBS_DIM];
    let mut next_obs = [0.0f32; OBS_DIM];

    let mut global_step = 0u64;
    let mut interval_steps = 0u64;
    let mut interval_start = Instant::now();

    for ep in 1..=args.episodes {
        env.reset();

        while !env.is_done() {
            env.get_observation(&mut obs);
            let action_idx = agent.select_action(&obs, epsilon, &mut rng);
            let action = headless_env::Action::wrap_index(action_idx);
            let (reward, done) = env.step(action);
            env.get_observation(&mut next_obs);

            buffer.push(Transition {
                state: obs,
                action: action_idx,
                reward,
                next_state: next_obs,
                done,
            });

            if global_step.is_multiple_of(4) && buffer.len() >= batch_size {
                let batch = buffer.sample(batch_size, &mut rng);
                let _ = agent.train_step(&batch, lr);
            }
            if global_step.is_multiple_of(100) {
                agent.sync_target();
            }
            global_step += 1;
            interval_steps += 1;
        }

        epsilon = (epsilon * gamma_decay).max(min_epsilon);

        if ep % 50 == 0 || ep == args.episodes {
            let secs = interval_start.elapsed().as_secs_f32().max(0.0001);
            let eps_per_sec = if ep.is_multiple_of(50) { 50.0 } else { (ep % 50) as f32 } / secs;
            let sps = interval_steps as f32 / secs;
            println!(
                "Ep {:5}/{} | eps {:.3} | SPS {:5.0} | EPS {:4.1} | score {}",
                ep,
                args.episodes,
                epsilon,
                sps,
                eps_per_sec,
                env.score
            );
            interval_steps = 0;
            interval_start = Instant::now();
        }
    }

    let total = start.elapsed();
    println!("Training complete in {total:.1?}");

    match agent.save_checkpoint(std::path::Path::new(&args.save)) {
        Ok(()) => println!("Saved policy to {}", args.save),
        Err(e) => println!("Failed to save checkpoint: {e:?}"),
    }
}
