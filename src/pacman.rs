//! The player: grid-based Pac-Man movement, input buffering, and pellet eating.
//!
//! Movement is discrete: a repeating [`MoveTimer`] ticks the player one tile
//! along its current [`Direction`]. A desired turn is *buffered* in
//! [`Player::queued`] and applied at the next tick if the target tile is free,
//! which gives the classic forgiving cornering feel without continuous motion.
//!
//! Eating is push-based: stepping onto a tile holding food removes it via
//! [`PelletMap::eat`] and triggers [`PelletEatenEvent`] or
//! [`PowerPelletEatenEvent`] for observers (score, frightened mode, win
//! detection all react downstream; nothing here reads or writes score).

#![allow(dead_code)]
#![allow(clippy::needless_pass_by_value)]

use crate::events::{
    Direction, GhostCollisionEvent, LifeLostEvent, PelletEatenEvent, PowerPelletEatenEvent,
};
use crate::ghosts::Ghost;
use crate::maze::{FoodEntities, MazeGrid, PelletKind, PelletMap, TILE_SIZE};
use bevy::prelude::*;

/// Seconds between player grid steps.
const MOVE_INTERVAL: f32 = 0.15;

/// Fill color of the player sprite (classic Pac-Man yellow).
const PLAYER_COLOR: Color = Color::srgb(1.0, 0.88, 0.19);

/// Z layer for the player sprite, above walls and pellets.
const Z_PLAYER: f32 = 0.5;

/// Fraction of [`TILE_SIZE`] that the player sprite spans.
const PLAYER_SCALE: f32 = 0.62;

/// Repeating cadence that advances the player one tile per tick.
///
/// Required on [`Player`]; every spawned player gets one automatically.
#[derive(Component, Debug, Deref, DerefMut)]
pub struct MoveTimer(Timer);

impl Default for MoveTimer {
    fn default() -> Self {
        Self(Timer::from_seconds(MOVE_INTERVAL, TimerMode::Repeating))
    }
}

/// Visual marker for the player sprite entity.
#[derive(Component, Debug, Default)]
pub struct PlayerTag;

/// The player pawn. `pos` is the source of truth; the sprite transform
/// follows it every tick.
#[derive(Component, Debug)]
#[require(Transform, Visibility, MoveTimer, PlayerTag)]
pub struct Player {
    /// Current grid tile.
    pub pos: IVec2,
    /// Heading used when no turn is queued.
    pub dir: Direction,
    /// Buffered turn request, consumed on the next successful step.
    pub queued: Option<Direction>,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            pos: IVec2::ZERO,
            dir: Direction::Left,
            queued: None,
        }
    }
}

/// Spawns the yellow player sprite at the maze's designated spawn tile.
pub fn spawn_player(mut commands: Commands, grid: Res<MazeGrid>) {
    let spawn = grid.pacman_spawn();
    let world = grid.world_pos(spawn);
    commands.spawn((
        Sprite::from_color(PLAYER_COLOR, Vec2::splat(TILE_SIZE * PLAYER_SCALE)),
        Transform::from_xyz(world.x, world.y, Z_PLAYER),
        Player {
            pos: spawn,
            ..Player::default()
        },
    ));
}

/// Reads WASD/arrow keys and buffers the newest requested direction.
///
/// Input only lands while playing; other states leave the buffer untouched
/// so a stale keypress cannot fire right after a reset.
pub fn handle_input(
    state: Res<State<crate::GameState>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut player: Query<&mut Player>,
) {
    if *state.get() != crate::GameState::Playing {
        return;
    }
    let Ok(mut player) = player.single_mut() else {
        return;
    };
    let requested = [
        ([KeyCode::KeyW, KeyCode::ArrowUp], Direction::Up),
        ([KeyCode::KeyS, KeyCode::ArrowDown], Direction::Down),
        ([KeyCode::KeyA, KeyCode::ArrowLeft], Direction::Left),
        ([KeyCode::KeyD, KeyCode::ArrowRight], Direction::Right),
    ]
    .into_iter()
    .find_map(|(codes, dir)| keyboard.any_pressed(codes).then_some(dir));
    if let Some(dir) = requested {
        player.queued = Some(dir);
    }
}

/// Advances the player one tile per [`MoveTimer`] tick and eats pellets.
///
/// A queued turn is tried first (only applied if its target tile is free);
/// otherwise the player keeps moving along `dir`. When both are blocked the
/// player stops in place while keeping its buffer, so steering into an open
/// corridor resumes movement without new input. An unconsumed queued turn
/// stays buffered across ticks until it becomes legal.
pub fn player_movement(
    time: Res<Time>,
    grid: Res<MazeGrid>,
    mut pellets: ResMut<PelletMap>,
    food: Res<FoodEntities>,
    mut query: Query<(Entity, &mut Player, &mut MoveTimer, &mut Transform)>,
    mut commands: Commands,
) {
    let Ok((entity, mut player, mut timer, mut transform)) = query.single_mut() else {
        return;
    };

    timer.tick(time.delta());
    if timer.just_finished()
        && let Some((next, heading)) =
            resolve_step(player.pos, player.dir, &mut player.queued, &grid)
    {
        player.dir = heading;
        player.pos = next;
        eat_at(&mut pellets, &food, &mut commands, entity, next);
    }

    let world = grid.world_pos(player.pos);
    transform.translation.x = world.x;
    transform.translation.y = world.y;
}

/// Resolves the next tile and heading for one grid step.
///
/// Tries the buffered turn first, falling back to the current heading;
/// returns `None` when both lead into walls. The buffer is cleared only when
/// actually consumed, so a blocked turn stays pending.
fn resolve_step(
    pos: IVec2,
    dir: Direction,
    queued: &mut Option<Direction>,
    grid: &MazeGrid,
) -> Option<(IVec2, Direction)> {
    if let Some(turned) = *queued {
        let candidate = pos + turned.to_delta();
        if !grid.is_wall(candidate) {
            *queued = None;
            return Some((candidate, turned));
        }
    }
    let forward = pos + dir.to_delta();
    if grid.is_wall(forward) {
        None
    } else {
        Some((forward, dir))
    }
}

/// Removes any food at `pos`, firing the matching event for observers.
fn eat_at(
    pellets: &mut PelletMap,
    food: &FoodEntities,
    commands: &mut Commands,
    _player: Entity,
    pos: IVec2,
) {
    if let Some(entity) = food.0.get(&pos) {
        commands.entity(*entity).despawn();
    }
    match pellets.eat(pos) {
        Some(PelletKind::Power) => {
            commands.trigger(PowerPelletEatenEvent { pos });
        }
        Some(PelletKind::Pellet) => {
            commands.trigger(PelletEatenEvent { pos });
        }
        None => {}
    }
}

/// Fires [`GhostCollisionEvent`] targeted at each ghost sharing the player's tile.
///
/// Runs every frame rather than on move ticks, so ghosts walking into a
/// stationary player are caught too. Observers decide lethality from the
/// ghost's mode (frightened ghosts are edible, not deadly).
pub fn check_ghost_collision(
    mut commands: Commands,
    players: Query<&Player>,
    ghosts: Query<(Entity, &Ghost)>,
) {
    let Ok(player) = players.single() else {
        return;
    };
    for (ghost_entity, ghost) in &ghosts {
        if ghost.pos == player.pos {
            commands.trigger(GhostCollisionEvent {
                ghost: ghost_entity,
            });
        }
    }
}


/// Observer: on life loss, returns Pac-Man to his spawn tile.
pub fn on_life_lost_reset(
    _trigger: On<LifeLostEvent>,
    grid: Res<MazeGrid>,
    mut player_q: Query<&mut Player, Without<crate::ghosts::Ghost>>,
) {
    let Ok(mut player) = player_q.single_mut() else {
        return;
    };
    player.pos = grid.pacman_spawn();
    player.dir = Direction::Left;
    player.queued = None;
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic)]

    use super::*;
    use std::collections::HashSet;

    fn grid() -> MazeGrid {
        MazeGrid::default()
    }

    #[test]
    fn default_player_faces_left_at_origin() {
        let player = Player::default();
        assert_eq!(player.pos, IVec2::ZERO);
        assert_eq!(player.dir, Direction::Left);
        assert_eq!(player.queued, None);
    }

    #[test]
    fn move_timer_defaults_to_repeating_150ms() {
        let MoveTimer(timer) = MoveTimer::default();
        assert_eq!(timer.mode(), TimerMode::Repeating);
        assert_eq!(timer.duration().as_secs_f32(), 0.15);
    }
    #[test]
    fn resolve_step_falls_back_to_heading_when_turn_blocked() {
        let g = grid();
        // From (1, 15): left is the border wall, right is an open corridor,
        // so the queued Left turn cannot fire and Right takes over.
        let mut queued = Some(Direction::Left);
        let step = resolve_step(IVec2::new(1, 15), Direction::Right, &mut queued, &g);
        assert_eq!(step, Some((IVec2::new(2, 15), Direction::Right)));
        assert_eq!(queued, Some(Direction::Left), "blocked turns stay queued");
    }
    #[test]
    fn resolve_step_is_none_when_both_paths_are_walls() {
        let g = grid();
        // At (1, 1) both left (border) and down (border below) are walls.
        let mut queued = Some(Direction::Down);
        let step = resolve_step(IVec2::new(1, 1), Direction::Left, &mut queued, &g);
        assert_eq!(step, None);
        assert_eq!(queued, Some(Direction::Down), "nothing consumed when blocked");
    }
    #[test]
    fn eating_a_dot_reports_pellet_then_exhausts_the_tile() {
        let mut pellets = PelletMap::from_maze(&grid());
        let before = pellets.remaining();
        assert_eq!(pellets.eat(IVec2::new(1, 15)), Some(PelletKind::Pellet));
        assert_eq!(pellets.remaining(), before - 1);
        assert_eq!(pellets.eat(IVec2::new(1, 15)), None, "tile must be exhausted");
    }
    #[test]
    fn eating_a_power_pellet_is_distinguished_from_a_dot() {
        let mut pellets = PelletMap::from_maze(&grid());
        assert_eq!(pellets.eat(IVec2::new(1, 13)), Some(PelletKind::Power));
        assert_eq!(pellets.eat(IVec2::new(1, 13)), None);
        assert_eq!(pellets.remaining_power(), 3, "one of four power pellets gone");
    }
    #[test]
    fn eating_spawn_and_wall_tiles_eats_nothing() {
        let g = grid();
        let mut pellets = PelletMap::from_maze(&g);
        assert_eq!(
            pellets.eat(g.pacman_spawn()),
            None,
            "spawn tiles hold no food"
        );
        assert_eq!(pellets.eat(IVec2::new(0, 0)), None, "walls hold no food");
        assert_eq!(
            pellets.eat(IVec2::new(-1, -1)),
            None,
            "out-of-bounds holds no food"
        );
    }

    #[test]
    fn player_spawn_tile_is_an_open_cell_in_the_real_maze() {
        let g = grid();
        assert!(!g.is_wall(g.pacman_spawn()));
        // Every neighbor of the spawn must not trap the player immediately.
        let open_neighbors = Direction::ALL
            .iter()
            .filter(|d| !g.is_wall(g.pacman_spawn() + d.to_delta()))
            .count();
        assert!(open_neighbors >= 2, "spawn needs room to maneuver");
    }

    #[test]
    fn every_food_position_from_maze_is_eatable_exactly_once() {
        let mut pellets = PelletMap::from_maze(&grid());
        let total = pellets.remaining();
        assert!(total > 0, "the real maze must contain food");
        // Re-eating every position twice must yield Some exactly once; collect
        // hits to confirm the count matches `remaining`.
        let mut hit_positions: HashSet<IVec2> = HashSet::new();
        for y in 0..crate::maze::HEIGHT as i32 {
            for x in 0..crate::maze::WIDTH as i32 {
                let pos = IVec2::new(x, y);
                if pellets.eat(pos).is_some() {
                    assert!(hit_positions.insert(pos), "double-reported {pos}");
                }
            }
        }
        assert_eq!(hit_positions.len(), total);
        assert_eq!(pellets.remaining(), 0, "board fully swept");
    }
}
