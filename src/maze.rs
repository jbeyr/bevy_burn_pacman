//! Pac-Man Lite maze: static ASCII layout, wall grid, and pellet tracking.
//!
//! Grid coordinates are `(x, y)` with `(0, 0)` at the bottom-left corner; `x`
//! grows right and `y` grows up. The [`MazeGrid`] resource parses the ASCII
//! blueprint into a wall grid and remembers the player/ghost spawn cells.
//! [`PelletMap`] tracks the food remaining on the board (win detection), and
//! [`spawn_maze`] renders walls and food as sprites that are cleaned up
//! automatically when the playing state exits.

#![allow(dead_code)]

use bevy::prelude::*;
use std::collections::HashSet;

// Rendering constants and some accessors await the DQN/AI phase.
/// World-space size of a single grid cell, in pixels.
pub const TILE_SIZE: f32 = 32.0;

/// Maze width in grid cells.
pub const WIDTH: usize = 21;

/// Maze height in grid cells.
pub const HEIGHT: usize = 17;

/// Fill color for wall tiles (classic dark-blue maze walls).
const WALL_COLOR: Color = Color::srgb(0.07, 0.10, 0.48);

/// Fill color for ordinary dot pellets.
const PELLET_COLOR: Color = Color::srgb(0.96, 0.82, 0.25);

/// Fill color for power pellets.
const POWER_PELLET_COLOR: Color = Color::srgb(0.98, 0.98, 0.98);

/// Fraction of [`TILE_SIZE`] that a dot pellet sprite spans.
const PELLET_SCALE: f32 = 0.25;

/// Fraction of [`TILE_SIZE`] that a power pellet sprite spans.
const POWER_PELLET_SCALE: f32 = 0.6;

/// Z layer for wall sprites.
const Z_WALL: f32 = 0.0;

/// Z layer for dot pellet sprites.
const Z_PELLET: f32 = 0.1;

/// Z layer for power pellet sprites.
const Z_POWER_PELLET: f32 = 0.2;

/// ASCII blueprint; the first row is drawn at the top of the screen.
///
/// `#` wall, `.` dot pellet, `o` power pellet, ` ` empty path,
/// `P` Pac-Man spawn, `G` ghost-house center. Blueprint row `i` maps to grid
/// row `y = HEIGHT - 1 - i`. The layout is a 21x17 classic-ish maze: a sealed
/// ghost house in the middle with a top door, side corridors connecting the
/// upper and lower halves, and four power pellets at the outer quadrants.
const MAZE_ASCII: [&str; HEIGHT] = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#o#.....#...#.....#o#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#####.#.###.#",
    "#...................#",
    "#####..### ###..#####",
    "#####..#  G  #..#####",
    "#####..#######..#####",
    "#.....#.......#.....#",
    "#.###.#.#####.#.###.#",
    "#o#...#.......#...#o#",
    "#.###.###.#.###.###.#",
    "#........P#.........#",
    "#####################",
];

/// Static maze wall layout, parsed from [`MAZE_ASCII`].
///
/// `is_wall` treats out-of-bounds cells as walls, so movement code never has
/// to branch on bounds separately.
#[derive(Resource, Debug, Clone)]
pub struct MazeGrid {
    /// Wall flags, indexed `walls[y][x]` with `y` growing upward.
    walls: [[bool; WIDTH]; HEIGHT],
    /// Cell where Pac-Man starts each level.
    pacman_spawn: IVec2,
    /// Cell at the center of the ghost house.
    ghost_spawn: IVec2,
}

impl Default for MazeGrid {
    fn default() -> Self {
        Self::parse_layout()
    }
}

impl MazeGrid {
    /// Parses [`MAZE_ASCII`] into a wall grid and locates the spawn cells.
    #[must_use]
    fn parse_layout() -> Self {
        let mut walls = [[false; WIDTH]; HEIGHT];
        let mut pacman_spawn = IVec2::new(0, 0);
        let mut ghost_spawn = IVec2::new(0, 0);
        for (row, line) in MAZE_ASCII.iter().enumerate() {
            let y = HEIGHT - 1 - row;
            for (x, &ch) in line.as_bytes().iter().enumerate() {
                if x >= WIDTH {
                    break;
                }
                walls[y][x] = ch == b'#';
                match ch {
                    b'P' => pacman_spawn = IVec2::new(x as i32, y as i32),
                    b'G' => ghost_spawn = IVec2::new(x as i32, y as i32),
                    _ => {}
                }
            }
        }
        Self {
            walls,
            pacman_spawn,
            ghost_spawn,
        }
    }

    /// Returns `true` when `pos` lies inside the grid.
    #[must_use]
    #[inline]
    #[allow(clippy::unused_self)]
    pub fn in_bounds(&self, pos: IVec2) -> bool {
        usize::try_from(pos.x).is_ok_and(|x| x < WIDTH)
            && usize::try_from(pos.y).is_ok_and(|y| y < HEIGHT)
    }

    /// Returns `true` when `pos` is a wall or lies outside the grid.
    #[must_use]
    #[inline]
    pub fn is_wall(&self, pos: IVec2) -> bool {
        let Ok(x) = usize::try_from(pos.x) else {
            return true;
        };
        let Ok(y) = usize::try_from(pos.y) else {
            return true;
        };
        x >= WIDTH || y >= HEIGHT || self.walls[y][x]
    }

    /// World position of a cell's center; the maze is centered on the origin,
    /// so the camera at the default position frames the whole board.
    ///
    /// `world_pos` does not depend on the wall data; it exists as a method so
    /// callers can address the grid uniformly.
    #[must_use]
    #[allow(clippy::unused_self)]
    pub const fn world_pos(&self, pos: IVec2) -> Vec3 {
        let offset = Vec2::new(
            WIDTH as f32 * TILE_SIZE * 0.5 - TILE_SIZE * 0.5,
            HEIGHT as f32 * TILE_SIZE * 0.5 - TILE_SIZE * 0.5,
        );
        Vec3::new(
            pos.x as f32 * TILE_SIZE - offset.x,
            pos.y as f32 * TILE_SIZE - offset.y,
            0.0,
        )
    }

    /// Cell where Pac-Man spawns each level.
    #[must_use]
    #[inline]
    pub const fn pacman_spawn(&self) -> IVec2 {
        self.pacman_spawn
    }

    /// Cell at the center of the ghost house.
    #[must_use]
    #[inline]
    pub const fn ghost_spawn(&self) -> IVec2 {
        self.ghost_spawn
    }
}

/// Kind of food a cell holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PelletKind {
    /// Ordinary dot pellet.
    Pellet,
    /// Power pellet that frightens the ghosts.
    Power,
}

/// Food remaining on the current level.
///
/// `remaining() == 0` means the level is cleared. The default value is an
/// empty board; build the real board with [`PelletMap::from_maze`].
#[derive(Resource, Debug, Default)]
pub struct PelletMap {
    /// Remaining dot pellet positions.
    pellets: HashSet<IVec2>,
    /// Remaining power pellet positions.
    power: HashSet<IVec2>,
}

impl PelletMap {
    /// Collects every pellet position from the maze layout, skipping wall
    /// cells so the map stays consistent with the wall grid.
    #[must_use]
    pub fn from_maze(grid: &MazeGrid) -> Self {
        let mut pellets = HashSet::new();
        let mut power = HashSet::new();
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let pos = IVec2::new(x as i32, y as i32);
                if grid.is_wall(pos) {
                    continue;
                }
                match MAZE_ASCII[HEIGHT - 1 - y].as_bytes()[x] {
                    b'.' => {
                        pellets.insert(pos);
                    }
                    b'o' => {
                        power.insert(pos);
                    }
                    _ => {}
                }
            }
        }
        Self { pellets, power }
    }

    /// Consumes the food at `pos`, returning its kind, or `None` if that cell
    /// was already eaten (or never held food).
    #[must_use]
    #[inline]
    pub fn eat(&mut self, pos: IVec2) -> Option<PelletKind> {
        if self.power.remove(&pos) {
            Some(PelletKind::Power)
        } else if self.pellets.remove(&pos) {
            Some(PelletKind::Pellet)
        } else {
            None
        }
    }

    /// Number of food items (dots + power pellets) still on the board.
    #[must_use]
    #[inline]
    pub fn remaining(&self) -> usize {
        self.pellets.len() + self.power.len()
    }

    /// Number of power pellets still on the board.
    #[must_use]
    #[inline]
    pub fn remaining_power(&self) -> usize {
        self.power.len()
    }

    /// Whether any food (dot or power) remains at `pos`.
    #[must_use]
    #[inline]
    pub fn has_food(&self, pos: IVec2) -> bool {
        self.pellets.contains(&pos) || self.power.contains(&pos)
    }

    /// Total food items that existed when the level started.
    ///
    /// The caller snapshots `remaining()` right after [`PelletMap::from_maze`];
    /// this helper exists for symmetry with `remaining`.
    #[must_use]
    pub fn total_initial(grid: &MazeGrid) -> usize {
        PelletMap::from_maze(grid).remaining()
    }

    /// Empties the board, e.g. when a level is cleared or reset.
    pub fn clear_level(&mut self) {
        self.pellets.clear();
        self.power.clear();
    }
}

/// Spawns wall, dot-pellet, and power-pellet sprites and (re)builds the
/// [`PelletMap`] resource.
///
/// Run once per level, e.g. from `OnEnter(GameState::Playing)` or at startup.
/// All spawned entities are `StateScoped(GameState::Playing)`, so leaving the
/// playing state removes them automatically.
pub fn spawn_maze(mut commands: Commands, grid: Res<MazeGrid>) {
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pos = IVec2::new(x as i32, y as i32);
            if grid.is_wall(pos) {
                let world = grid.world_pos(pos);
                commands.spawn((
                    Sprite::from_color(WALL_COLOR, Vec2::splat(TILE_SIZE)),
                    Transform::from_xyz(world.x, world.y, Z_WALL),
                    
                ));
            }
        }
    }
    let pellets = PelletMap::from_maze(&grid);
    for &pos in &pellets.pellets {
        let world = grid.world_pos(pos);
        commands.spawn((
            Sprite::from_color(PELLET_COLOR, Vec2::splat(TILE_SIZE * PELLET_SCALE)),
            Transform::from_xyz(world.x, world.y, Z_PELLET),
            
        ));
    }
    for &pos in &pellets.power {
        let world = grid.world_pos(pos);
        commands.spawn((
            Sprite::from_color(
                POWER_PELLET_COLOR,
                Vec2::splat(TILE_SIZE * POWER_PELLET_SCALE),
            ),
            Transform::from_xyz(world.x, world.y, Z_POWER_PELLET),
            
        ));
    }
    commands.insert_resource(pellets);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic)]

    use super::*;
    use std::collections::VecDeque;

    /// Every blueprint row is exactly `WIDTH` characters wide, so parsing
    /// never leaves implicit open cells at row edges.
    #[test]
    fn layout_rows_are_all_full_width() {
        for (row, line) in MAZE_ASCII.iter().enumerate() {
            assert_eq!(line.len(), WIDTH, "blueprint row {row} has wrong width");
        }
    }

    /// Walls, openings, spawns, and out-of-bounds behavior parse correctly.
    #[test]
    fn parsing_places_walls_and_openings() {
        let grid = MazeGrid::default();
        // Corners are walls.
        assert!(grid.is_wall(IVec2::new(0, 0)));
        assert!(grid.is_wall(IVec2::new(WIDTH as i32 - 1, 0)));
        assert!(grid.is_wall(IVec2::new(0, HEIGHT as i32 - 1)));
        // Out-of-bounds cells count as walls but are not in bounds.
        assert!(!grid.in_bounds(IVec2::new(-1, 0)));
        assert!(grid.is_wall(IVec2::new(-1, 0)));
        assert!(!grid.in_bounds(IVec2::new(0, HEIGHT as i32)));
        // The door above the ghost house is open.
        assert!(!grid.is_wall(IVec2::new(10, 8)));
        // Spawn cells are open.
        assert!(!grid.is_wall(grid.pacman_spawn()));
        assert!(!grid.is_wall(grid.ghost_spawn()));
        // Sanity: 21x17 grid with 169 open cells (357 - 188 walls).
        let open = (0..HEIGHT)
            .flat_map(|y| (0..WIDTH).map(move |x| IVec2::new(x as i32, y as i32)))
            .filter(|pos| !grid.is_wall(*pos))
            .count();
        assert_eq!(open, 169);
    }

    /// The spawn cells match the `P`/`G` markers in the blueprint.
    #[test]
    fn spawn_positions_match_ascii_layout() {
        let grid = MazeGrid::default();
        assert_eq!(grid.pacman_spawn(), IVec2::new(9, 1));
        assert_eq!(grid.ghost_spawn(), IVec2::new(10, 7));
    }

    /// `from_maze` counts every dot and power pellet in the blueprint.
    #[test]
    fn from_maze_counts_all_food() {
        let grid = MazeGrid::default();
        let pellets = PelletMap::from_maze(&grid);
        assert_eq!(pellets.remaining(), 162);
        assert_eq!(pellets.remaining_power(), 4);
    }

    /// `eat` removes food and reports its kind exactly once per cell.
    #[test]
    fn eat_consumes_pellets_and_reports_kind() {
        let grid = MazeGrid::default();
        let mut pellets = PelletMap::from_maze(&grid);
        // A plain dot next to the Pac-Man spawn.
        assert_eq!(pellets.eat(IVec2::new(1, 1)), Some(PelletKind::Pellet));
        assert_eq!(pellets.remaining(), 161);
        // Power pellets sit at the outer corners of rows 3 and 13.
        assert_eq!(pellets.eat(IVec2::new(1, 3)), Some(PelletKind::Power));
        assert_eq!(pellets.remaining_power(), 3);
        // Eating an already-empty cell reports nothing.
        assert_eq!(pellets.eat(IVec2::new(1, 1)), None);
        assert_eq!(pellets.eat(IVec2::new(10, 7)), None); // ghost house center
        assert_eq!(pellets.eat(IVec2::new(-5, -5)), None); // out of bounds
    }

    /// `clear_level` empties every food set.
    #[test]
    fn clear_level_resets_all_food() {
        let grid = MazeGrid::default();
        let mut pellets = PelletMap::from_maze(&grid);
        pellets.clear_level();
        assert_eq!(pellets.remaining(), 0);
        assert_eq!(pellets.remaining_power(), 0);
        assert_eq!(pellets.eat(IVec2::new(1, 1)), None);
    }

    /// Every open cell — and therefore every pellet — is reachable from the
    /// Pac-Man spawn, so win detection can never stall on unreachable food.
    #[test]
    fn every_open_cell_is_reachable_from_pacman_spawn() {
        let grid = MazeGrid::default();
        let pellets = PelletMap::from_maze(&grid);
        let start = grid.pacman_spawn();
        let mut reachable = HashSet::from([start]);
        let mut queue = VecDeque::from([start]);
        let deltas = [
            IVec2::new(1, 0),
            IVec2::new(-1, 0),
            IVec2::new(0, 1),
            IVec2::new(0, -1),
        ];
        while let Some(pos) = queue.pop_front() {
            for delta in deltas {
                let next = pos + delta;
                if grid.in_bounds(next) && !grid.is_wall(next) && reachable.insert(next) {
                    queue.push_back(next);
                }
            }
        }
        assert_eq!(reachable.len(), 169);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let pos = IVec2::new(x as i32, y as i32);
                if !grid.is_wall(pos) {
                    assert!(reachable.contains(&pos), "unreachable cell {pos:?}");
                }
            }
        }
        assert_eq!(pellets.remaining(), 162);
    }

    /// `world_pos` maps the maze center to the origin and cell corners to the
    /// maze extents.
    #[test]
    fn world_pos_centers_maze_on_origin() {
        let grid = MazeGrid::default();
        assert_eq!(grid.world_pos(IVec2::new(10, 8)), Vec3::ZERO);
        assert_eq!(
            grid.world_pos(IVec2::new(0, 0)),
            Vec3::new(-320.0, -256.0, 0.0)
        );
        assert_eq!(
            grid.world_pos(IVec2::new(20, 16)),
            Vec3::new(320.0, 256.0, 0.0)
        );
    }
}
