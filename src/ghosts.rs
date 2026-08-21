//! Scripted ghost adversaries: chase / frightened / eaten state machine.
//!
//! Ghosts share the player's grid cadence via a per-ghost [`StepTimer`]
//! component. Each [`GhostKind`] has a distinct targeting heuristic (classic
//! Pac-Man personalities). On power-pellet consumption all non-eaten ghosts
//! switch to [`GhostMode::Frightened`] and move randomly; eaten ghosts return
//! to the ghost house before respawning into chase mode. Frightened movement
//! uses a fixed-seed [`StdRng`] injected through a [`Local`], so ghost
//! behavior is deterministic for a given seed.

#![allow(dead_code)]
#![allow(clippy::needless_pass_by_value)]

use bevy::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::events::{Direction, LifeLostEvent, PowerPelletEatenEvent};
use crate::maze::{MazeGrid, PelletMap, HEIGHT, TILE_SIZE};
use crate::pacman::Player;

// Eaten-mode plumbing activates with the DQN phase.
/// Fixed seed for the frightened-mode RNG, keeping ghost behavior reproducible.
const FRIGHT_RNG_SEED: u64 = 0x0F1F_71ED;

/// Z layer for ghost sprites: above walls and pellets.
const GHOST_Z: f32 = 0.3;

/// Fraction of [`TILE_SIZE`] a ghost sprite spans.
const GHOST_SCALE: f32 = 0.8;

/// All four directions; the reverse of a ghost's heading is filtered at
/// decision time, so this mirrors `Direction`'s variants.
const ALL_DIRECTIONS: [Direction; 4] = [
    Direction::Up,
    Direction::Down,
    Direction::Left,
    Direction::Right,
];

/// Ghost personality, each with a distinct targeting heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhostKind {
    /// Targets Pac-Man's tile directly.
    Blinky,
    /// Targets four tiles ahead of Pac-Man.
    Pinky,
    /// Targets the point mirrored through Pac-Man from this ghost.
    Inky,
    /// Targets Pac-Man when far, retreats to his corner when close.
    Clyde,
}

impl GhostKind {
    /// All four ghosts in spawn order.
    pub const ALL: [Self; 4] = [Self::Blinky, Self::Pinky, Self::Inky, Self::Clyde];

    /// Display color for the sprite.
    #[must_use]
    pub const fn color(self) -> Color {
        match self {
            Self::Blinky => Color::srgb(0.9, 0.15, 0.15),
            Self::Pinky => Color::srgb(1.0, 0.65, 0.85),
            Self::Inky => Color::srgb(0.2, 0.9, 0.9),
            Self::Clyde => Color::srgb(0.95, 0.6, 0.15),
        }
    }

    /// Classic targeting heuristic: the tile this ghost wants to reach.
    #[must_use]
    pub fn target_heuristic(self, pac_pos: IVec2, pac_dir: Direction, self_pos: IVec2) -> IVec2 {
        match self {
            Self::Blinky => pac_pos,
            Self::Pinky => pac_pos + pac_dir.to_delta() * 4,
            Self::Inky => pac_pos * 2 - self_pos,
            Self::Clyde => {
                if manhattan(pac_pos, self_pos) > 8 {
                    pac_pos
                } else {
                    IVec2::new(-8, HEIGHT as i32)
                }
            }
        }
    }
}

/// Manhattan distance between two grid cells.
#[must_use]
fn manhattan(a: IVec2, b: IVec2) -> i32 {
    let delta = (a - b).abs();
    delta.x + delta.y
}

/// Behavioral state of a ghost.
#[derive(Debug, Clone)]
pub enum GhostMode {
    /// Actively hunting Pac-Man using the kind heuristic.
    Chase,
    /// Fleeing after a power pellet; moves randomly.
    Frightened { timer: Timer },
    /// Returning to the ghost house after being eaten.
    Eaten { timer: Timer },
}

/// Repeating cadence that advances a ghost one tile per tick.
#[derive(Component, Debug, Deref, DerefMut)]
pub struct StepTimer(Timer);

impl StepTimer {
    /// A fresh repeating timer, staggered by `offset` seconds so the pack
    /// does not move in lockstep from frame zero.
    #[must_use]
    fn new(offset: f32) -> Self {
        Self(Timer::from_seconds(
            Ghost::STEP_SECS + offset,
            TimerMode::Repeating,
        ))
    }
}

impl Default for StepTimer {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Visual marker for ghost sprite entities.
#[derive(Component, Debug)]
pub struct GhostTag;

/// A scripted ghost adversary.
#[derive(Component, Debug)]
#[require(Transform, Visibility, StepTimer)]
pub struct Ghost {
    /// Personality driving the targeting heuristic.
    pub kind: GhostKind,
    /// Current grid tile; the sprite transform follows this.
    pub pos: IVec2,
    /// Heading used at the next step.
    pub dir: Direction,
    /// Behavioral state (chase / frightened / eaten).
    pub mode: GhostMode,
}

impl Ghost {
    /// Frightened duration after a power pellet.
    pub const FRIGHTENED_SECS: f32 = 7.0;
    /// Time spent returning to the house after being eaten.
    pub const EATEN_SECS: f32 = 3.0;
    /// Seconds per grid step.
    pub const STEP_SECS: f32 = 0.18;
}

/// Deterministic RNG for frightened movement.
///
/// A single fixed seed keeps ghost behavior reproducible: no global RNG
/// state, no per-frame re-seeding. [`FromWorld`] lets Bevy construct the
/// value for [`Local`] injection into [`ghost_movement`].
#[derive(Debug)]
pub struct FrightRng(StdRng);

impl FromWorld for FrightRng {
    fn from_world(_world: &mut World) -> Self {
        Self(StdRng::seed_from_u64(FRIGHT_RNG_SEED))
    }
}

/// Spawns the four ghosts at the ghost-house tile with distinct colors.
pub fn spawn_ghosts(mut commands: Commands, grid: Res<MazeGrid>) {
    let spawn = grid.ghost_spawn();
    let world = grid.world_pos(spawn);
    for (i, kind) in GhostKind::ALL.into_iter().enumerate() {
        commands.spawn((
            Sprite::from_color(kind.color(), Vec2::splat(TILE_SIZE * GHOST_SCALE)),
            Transform::from_xyz(world.x, world.y, GHOST_Z),
            Ghost {
                kind,
                pos: spawn,
                dir: Direction::Left,
                mode: GhostMode::Chase,
            },
            GhostTag,
            StepTimer::new((i as f32) * 0.02),
        ));
    }
}
/// Advances each ghost one grid step on its own [`StepTimer`].
///
/// `pellets_unused` is read-only: ghosts never eat, but the board gates
/// movement so the pack stands down once the level is cleared.
pub fn ghost_movement(
    time: Res<Time>,
    grid: Res<MazeGrid>,
    pellets_unused: Res<PelletMap>,
    pac_query: Query<&Player>,
    mut ghost_q: Query<(Entity, &mut Ghost, &mut StepTimer, &mut Transform)>,
    mut commands: Commands,
    mut rng: Local<FrightRng>,
) {
    if pellets_unused.remaining() == 0 {
        return; // board cleared: the level is over, ghosts stand down
    }
    let Ok(player) = pac_query.single() else {
        return;
    };
    for (entity, mut ghost, mut step, mut transform) in &mut ghost_q {
        step.tick(time.delta());
        if !step.just_finished() {
            continue;
        }

        let returning = matches!(ghost.mode, GhostMode::Eaten { .. });
        if returning && ghost.pos == grid.ghost_spawn() {
            // Home: pause a beat, then resume hunting.
            ghost.mode = GhostMode::Chase;
            commands.entity(entity).insert(StepTimer::default());
            continue;
        }

        let next_dir = choose_direction(&grid, &ghost, player.pos, returning, &mut rng.0);
        ghost.dir = next_dir;
        let candidate = ghost.pos + next_dir.to_delta();
        if !grid.is_wall(candidate) {
            ghost.pos = candidate;
        }
        let world = grid.world_pos(ghost.pos);
        transform.translation.x = world.x;
        transform.translation.y = world.y;
    }
}

/// Picks the next direction according to the current mode.
///
/// Reversing is only allowed when the ghost is in a dead end; otherwise the
/// ghost keeps its heading or turns onto an open tile.
fn choose_direction(
    grid: &MazeGrid,
    ghost: &Ghost,
    pac_pos: IVec2,
    returning: bool,
    rng: &mut StdRng,
) -> Direction {
    let reverse = ghost.dir.opposite();
    let forward_blocked = grid.is_wall(ghost.pos + ghost.dir.to_delta());
    let mut candidates: Vec<Direction> = ALL_DIRECTIONS
        .into_iter()
        .filter(|d| *d != reverse || forward_blocked)
        .filter(|d| !grid.is_wall(ghost.pos + d.to_delta()))
        .collect();

    if candidates.is_empty() {
        return reverse;
    }

    if returning {
        // Head home deterministically: minimize distance to the house.
        candidates.sort_by_key(|d| manhattan(ghost.pos + d.to_delta(), grid.ghost_spawn()));
        return candidates.first().copied().unwrap_or(reverse);
    }

    match &ghost.mode {
        GhostMode::Frightened { .. } => {
            let idx = rng.gen_range(0..candidates.len());
            candidates.swap_remove(idx)
        }
        GhostMode::Chase | GhostMode::Eaten { .. } => {
            let target = ghost.kind.target_heuristic(pac_pos, ghost.dir, ghost.pos);
            candidates.sort_by_key(|d| manhattan(ghost.pos + d.to_delta(), target));
            candidates.first().copied().unwrap_or(reverse)
        }
    }
}

/// Observer: power pellets frighten every non-eaten ghost.
pub fn frighten_all(_trigger: On<PowerPelletEatenEvent>, mut ghost_q: Query<&mut Ghost>) {
    for mut ghost in &mut ghost_q {
        if !matches!(ghost.mode, GhostMode::Eaten { .. }) {
            ghost.mode = GhostMode::Frightened {
                timer: Timer::from_seconds(Ghost::FRIGHTENED_SECS, TimerMode::Once),
            };
        }
    }
}

/// Ticks frightened/eaten timers every frame and reverts to chase when done.
pub fn update_mode_timers(time: Res<Time>, mut ghost_q: Query<&mut Ghost>) {
    for mut ghost in &mut ghost_q {
        let finished = match &mut ghost.mode {
            GhostMode::Frightened { timer } | GhostMode::Eaten { timer } => {
                timer.tick(time.delta());
                timer.is_finished()
            }
            GhostMode::Chase => false,
        };
        if finished {
            ghost.mode = GhostMode::Chase;
        }
    }
}
/// Observer: on life loss, returns every ghost to the house in chase mode.
pub fn on_life_lost_reset(
    _trigger: On<LifeLostEvent>,
    grid: Res<MazeGrid>,
    mut ghost_q: Query<&mut Ghost>,
) {
    let house = grid.ghost_spawn();
    for mut ghost in &mut ghost_q {
        ghost.pos = house;
        ghost.dir = Direction::Left;
        ghost.mode = GhostMode::Chase;
    }
}
