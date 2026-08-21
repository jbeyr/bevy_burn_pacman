//! Neural autopilot: loads the trained Double DQN policy and drives Pac-Man.
//!
//! The policy was trained in the headless [`crate::headless_env`]; this module
//! rebuilds the identical 20-float observation from live ECS state each frame
//! and writes the argmax action into [`Player::queued`] on the movement cadence.
//! Toggle with `A`; the banner text shows which brain is driving.

use bevy::prelude::*;
use bevy::state::state_scoped::DespawnOnExit as StateScoped;
use bevy::text::FontSize;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_ndarray::{NdArray, NdArrayDevice};

use crate::agent::PacQNetwork;
use crate::events::Direction;
use crate::headless_env;
use crate::maze::PelletMap;
use crate::pacman::Player;
use crate::ghosts::Ghost;

type NdArrayF32 = NdArray<f32>;

/// The loaded network plus its device, or `None` when no checkpoint exists.
#[derive(Resource)]
pub struct PolicyResource {
    pub net: Option<PacQNetwork<NdArrayF32>>,
    pub device: NdArrayDevice,
}

/// Whether the autopilot is currently steering.
#[derive(Debug, Resource, Default)]
pub struct Autopilot(pub bool);

/// Marker for the HUD status line.
#[derive(Component, Debug)]
pub struct AutopilotText;

/// Plugin: resource loading, toggle input, per-frame inference, HUD badge.
#[derive(Debug)]
pub struct AutopilotPlugin {
    /// Checkpoint path passed through from the CLI.
    pub checkpoint: std::path::PathBuf,
}

impl Plugin for AutopilotPlugin {
    fn build(&self, app: &mut App) {
        let device = NdArrayDevice::default();
        let mut net = None;

        if self.checkpoint.exists() {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
            match recorder.load(self.checkpoint.clone(), &device) {
                Ok(record) => {
                    let untrained = PacQNetwork::<NdArrayF32>::new(&device);
                    net = Some(untrained.load_record(record));
                    println!(
                        "Policy loaded from {} — press A to toggle autopilot.",
                        self.checkpoint.display()
                    );
                }
                Err(e) => println!("Failed to load {}: {e:?}", self.checkpoint.display()),
            }
        } else {
            println!(
                "No checkpoint at {} — training one first: cargo run --release -- --headless",
                self.checkpoint.display()
            );
        }

        app.insert_resource(PolicyResource { net, device })
            .init_resource::<Autopilot>()
            .add_systems(Startup, spawn_badge)
            .add_systems(Update, toggle_key)
            .add_systems(
                Update,
                (drive_player, update_badge).run_if(in_state(crate::GameState::Playing)),
            );
    }
}

fn spawn_badge(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(8.0),
            left: Val::Px(16.0),
            ..default()
        },
        Text::new("[A] Autopilot: OFF"),
        TextFont { font_size: FontSize::Px(18.0), ..default() },
        TextColor(Color::srgb(0.55, 0.75, 1.0)),
        StateScoped(crate::GameState::Playing),
        AutopilotText,
    ));
}

fn toggle_key(keys: Res<ButtonInput<KeyCode>>, mut autopilot: ResMut<Autopilot>) {
    if keys.just_pressed(KeyCode::KeyA) {
        autopilot.0 = !autopilot.0;
    }
}

/// Rebuilds the headless observation vector from live ECS state.
///
/// Must stay byte-compatible with `PacEnv::get_observation` or the policy acts
/// on a world it never trained on.
fn build_observation(
    player: &Player,
    grid: &crate::maze::MazeGrid,
    pellets: &PelletMap,
    ghosts: &[(IVec2, crate::ghosts::GhostMode)],
    fright_active: bool,
    remaining_frac: f32,
    out: &mut [f32; headless_env::OBS_DIM],
) {
    out.fill(0.0);

    let dirs = [(0i32, 1i32), (0, -1), (-1, 0), (1, 0)];
    for (i, (dx, dy)) in dirs.into_iter().enumerate() {
        let next = IVec2::new(player.pos.x + dx, player.pos.y + dy);
        out[i] = f32::from(u8::from(grid.is_wall(next)));
        out[4 + i] = f32::from(u8::from(pellets.has_food(next)));
    }

    for (g_pos, g_mode) in ghosts {
        let dx = g_pos.x - player.pos.x;
        let dy = g_pos.y - player.pos.y;
        let dist = (dx.abs() + dy.abs()).max(1);
        let scale = 1.0 / dist as f32;
        let threat = if matches!(g_mode, crate::ghosts::GhostMode::Frightened { .. }) { -scale } else { scale };
        if dx.abs() >= dy.abs() {
            out[8 + usize::from(dx > 0) * 2 + usize::from(dx < 0)] += threat;
        } else {
            out[9 + usize::from(dy < 0) * 2 + usize::from(dy > 0)] += threat;
        }
    }

    out[12] = f32::from(u8::from(fright_active));
    out[13] = remaining_frac;

    let face_idx = match player.dir {
        Direction::Up => 15,
        Direction::Down => 16,
        Direction::Left => 17,
        Direction::Right => 18,
    };
    out[face_idx] = 1.0;
}

fn drive_player(
    autopilot: Res<Autopilot>,
    policy: Res<PolicyResource>,
    mut player_q: Query<&mut Player>,
    grid: Res<crate::maze::MazeGrid>,
    pellets: Res<PelletMap>,
    ghost_q: Query<&Ghost>,
) {
    if !autopilot.0 {
        return;
    }
    let Some(net) = &policy.net else {
        return;
    };
    let Ok(mut player) = player_q.single_mut() else {
        return;
    };

    let fright_active =
        ghost_q.iter().any(|g| matches!(g.mode, crate::ghosts::GhostMode::Frightened { .. }));
    let remaining_frac = f32::from(u8::from(pellets.remaining() > 0)) * 0.999;

    let mut obs = [0.0f32; headless_env::OBS_DIM];
    let ghost_snapshot: Vec<(IVec2, crate::ghosts::GhostMode)> =
        ghost_q.iter().map(|g| (g.pos, g.mode.clone())).collect();
    build_observation(&player, &grid, &pellets, &ghost_snapshot, fright_active, remaining_frac, &mut obs);

    let data = burn::tensor::TensorData::new(obs.to_vec(), [1, headless_env::OBS_DIM]);
    let input = burn::tensor::Tensor::<NdArrayF32, 2>::from_data(data, &policy.device);
    let q = net.forward(input).into_data();
    let slice = q.as_slice::<f32>().unwrap_or(&[]);

    let best = (0..4)
        .max_by(|a, b| {
            slice
                .get(*a)
                .copied()
                .unwrap_or(0.0)
                .total_cmp(&slice.get(*b).copied().unwrap_or(0.0))
        })
        .unwrap_or(0);

    if let Some(action) = headless_env::Action::from_index(best) {
        player.queued = Some(match action {
            headless_env::Action::Up => Direction::Up,
            headless_env::Action::Down => Direction::Down,
            headless_env::Action::Left => Direction::Left,
            headless_env::Action::Right => Direction::Right,
        });
    }
}

fn update_badge(autopilot: Res<Autopilot>, policy: Res<PolicyResource>, mut q: Query<&mut Text, With<AutopilotText>>) {
    let Ok(mut text) = q.single_mut() else {
        return;
    };
    let state = match (&policy.net, autopilot.0) {
        (None, _) => "[A] Autopilot unavailable (no checkpoint)",
        (_, false) => "[A] Autopilot: OFF",
        (_, true) => "[A] Autopilot: ON",
    };
    **text = state.to_string();
}
