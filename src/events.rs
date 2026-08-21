//! Shared directions and push-based observer events for Pac-Man Lite.
//!
//! Bevy 0.19 uses a push-based event model: events are *triggered* with
//! [`Commands::trigger`] and handled by observers registered with `On<...>`
//! (e.g. `commands.add_observer(|event: On<PelletEatenEvent>| ...)`).
//! There is no `EventWriter`/`EventReader` or `add_event` registration anymore.
//!
//! [`GhostCollisionEvent`] is an [`EntityEvent`]: it carries a target entity
//! (the ghost that touched the player), so both global observers and
//! entity-scoped observers attached to that ghost observe it.

#![allow(dead_code)]

use bevy::prelude::*;

// Some contract items await the DQN phase; keep them compiled and documented.
/// One of the four cardinal directions on the grid.
///
/// [`Direction::to_delta`] and [`Direction::from_ivec2`] are inverse
/// conversions between directions and unit grid displacements; [`Direction::ALL`]
/// iterates in a stable order (`Up, Down, Left, Right`) shared with the DQN
/// action head in a later phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    /// All four directions in stable order.
    pub const ALL: [Self; 4] = [Self::Up, Self::Down, Self::Left, Self::Right];

    /// Unit displacement in grid coordinates (`+y` is up).
    #[must_use]
    pub const fn to_delta(self) -> IVec2 {
        match self {
            Self::Up => IVec2::new(0, 1),
            Self::Down => IVec2::new(0, -1),
            Self::Left => IVec2::new(-1, 0),
            Self::Right => IVec2::new(1, 0),
        }
    }

    /// The direction pointing the opposite way.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Down => Self::Up,
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }

    /// The unique cardinal direction whose [`Direction::to_delta`] equals
    /// `delta`, or `None` if `delta` is not a unit cardinal step.
    #[must_use]
    pub const fn from_ivec2(delta: IVec2) -> Option<Self> {
        match (delta.x, delta.y) {
            (0, 1) => Some(Self::Up),
            (0, -1) => Some(Self::Down),
            (-1, 0) => Some(Self::Left),
            (1, 0) => Some(Self::Right),
            _ => None,
        }
    }
}

/// Fired when the player steps onto a regular pellet tile.
#[derive(Event, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PelletEatenEvent {
    /// Grid position of the eaten pellet.
    pub pos: IVec2,
}

/// Fired when the player steps onto a power pellet tile.
///
/// Observers switch ghosts to frightened mode and award the pickup score.
#[derive(Event, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerPelletEatenEvent {
    /// Grid position of the eaten power pellet.
    pub pos: IVec2,
}

/// Fired when the player shares a tile with a ghost.
///
/// This is an [`EntityEvent`] targeted at the ghost entity: observers may be
/// global or scoped to that ghost. Whether the hit is lethal or merely sends
/// the ghost home depends on the ghost's mode at observation time.
#[derive(EntityEvent, Debug, Clone, Copy, PartialEq, Eq)]
pub struct GhostCollisionEvent {
    /// The ghost entity that collided with the player.
    #[event_target]
    pub ghost: Entity,
}

/// Fired when the last pellet/power pellet is eaten.
#[derive(Event, Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelClearedEvent;

/// Fired when the player is caught and loses one of its lives.
#[derive(Event, Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifeLostEvent;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deltas_are_unit_cardinals() {
        assert_eq!(Direction::Up.to_delta(), IVec2::new(0, 1));
        assert_eq!(Direction::Down.to_delta(), IVec2::new(0, -1));
        assert_eq!(Direction::Left.to_delta(), IVec2::new(-1, 0));
        assert_eq!(Direction::Right.to_delta(), IVec2::new(1, 0));
    }

    #[test]
    fn from_ivec2_roundtrips_every_direction() {
        for dir in Direction::ALL {
            assert_eq!(Direction::from_ivec2(dir.to_delta()), Some(dir));
        }
    }

    #[test]
    fn from_ivec2_rejects_non_cardinal_deltas() {
        assert_eq!(Direction::from_ivec2(IVec2::ZERO), None);
        assert_eq!(Direction::from_ivec2(IVec2::new(2, 0)), None);
        assert_eq!(Direction::from_ivec2(IVec2::new(1, 1)), None);
        assert_eq!(Direction::from_ivec2(IVec2::new(-1, -1)), None);
        assert_eq!(Direction::from_ivec2(IVec2::new(0, -2)), None);
    }

    #[test]
    fn opposite_is_involutive_and_distinct() {
        for dir in Direction::ALL {
            assert_eq!(dir.opposite().opposite(), dir);
            assert_ne!(dir.opposite(), dir);
        }
    }

    #[test]
    fn opposite_deltas_cancel() {
        for dir in Direction::ALL {
            assert_eq!(dir.to_delta() + dir.opposite().to_delta(), IVec2::ZERO);
        }
    }

    #[test]
    fn all_lists_exactly_four_distinct_directions() {
        assert_eq!(Direction::ALL.len(), 4);
        for (i, a) in Direction::ALL.iter().enumerate() {
            for b in &Direction::ALL[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
