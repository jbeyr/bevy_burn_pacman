//! Deterministic headless Pac-Man environment mirroring the Bevy game rules.
//!
//! Shares the maze blueprint semantics with `maze.rs` but runs without any ECS,
//! so thousands of episodes can train per second. Ghosts use the same
//! heuristics as the visual ghosts (direct chase, frightened flee).

use crate::events::Direction;
use crate::maze::{MazeGrid, PelletKind, PelletMap};
use bevy::math::IVec2;

/// Actions available to the learning agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    /// Number of discrete actions.
    pub const COUNT: usize = 4;

    /// All actions.
    pub const ALL: [Self; Self::COUNT] = [Self::Up, Self::Down, Self::Left, Self::Right];

    /// Converts an index into an action.
    #[must_use]
    pub const fn wrap_index(index: usize) -> Self {
        match index % Self::COUNT {
            0 => Self::Up,
            1 => Self::Down,
            2 => Self::Left,
            _ => Self::Right,
        }
    }

    /// Converts an index into an action, `None` when out of range.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        if index < Self::COUNT { Some(Self::wrap_index(index)) } else { None }
    }

    /// The action's index for network heads.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Grid delta for this action.
    #[must_use]
    pub const fn to_delta(self) -> i32 {
        match self {
            Self::Up => 0,
            Self::Down => 1,
            Self::Left => 2,
            Self::Right => 3,
        }
    }

    /// IVec2 delta of this action.
    #[must_use]
    pub const fn delta(self) -> (i32, i32) {
        match self {
            Self::Up => (0, 1),
            Self::Down => (0, -1),
            Self::Left => (-1, 0),
            Self::Right => (1, 0),
        }
    }
}

/// One scripted ghost in the headless world.
#[derive(Debug, Clone)]
pub struct HeadlessGhost {
    pub x: i32,
    pub y: i32,
    pub frightened_steps: u32,
    pub rng_state: u64,
}

impl HeadlessGhost {
    fn next_random(&mut self) -> u64 {
        // xorshift64: deterministic, dependency-free
        let mut s = self.rng_state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.rng_state = s;
        s
    }
}

/// Observation vector dimensionality.
pub const OBS_DIM: usize = 20;

/// Full deterministic game state for headless training.
#[derive(Debug)]
pub struct PacEnv {
    grid: MazeGrid,
    pellets: PelletMap,
    /// Mirror of remaining pellet positions (PelletMap fields are private).
    pellet_mirror: std::collections::HashSet<IVec2>,
    total_pellets: usize,
    pub px: i32,
    pub py: i32,
    pub facing: Direction,
    pub ghosts: Vec<HeadlessGhost>,
    pub steps: usize,
    pub max_steps: usize,
    pub score: u32,
    pub lives: u32,
    done: bool,
}

impl Default for PacEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl PacEnv {
    /// Creates a fresh environment with 4 chase ghosts and a 600-step horizon.
    #[must_use]
    pub fn new() -> Self {
        let grid = MazeGrid::default();
        let pellet_mirror = initial_pellets(&grid);
        let total_pellets = pellet_mirror.len();
        let mut env = Self {
            pellets: PelletMap::from_maze(&grid),
            pellet_mirror,
            total_pellets,
            grid,
            px: 0,
            py: 0,
            facing: Direction::Left,
            ghosts: Vec::new(),
            steps: 0,
            max_steps: 600,
            score: 0,
            lives: 3,
            done: false,
        };
        env.reset_positions();
        env
    }

    fn reset_positions(&mut self) {
        let spawn = self.grid.pacman_spawn();
        self.px = spawn.x;
        self.py = spawn.y;
        self.facing = Direction::Left;
        let house = self.grid.ghost_spawn();
        self.ghosts = (0..4)
            .map(|i| HeadlessGhost {
                x: house.x + i - 1,
                y: house.y,
                frightened_steps: 0,
                rng_state: 0x5EED_0000 + i as u64,
            })
            .collect();
    }

    /// Restarts the episode fully (pellets restored).
    pub fn reset(&mut self) {
        self.pellets = PelletMap::from_maze(&self.grid);
        self.pellet_mirror = initial_pellets(&self.grid);
        self.steps = 0;
        self.score = 0;
        self.lives = 3;
        self.done = false;
        self.reset_positions();
    }

    /// Whether Pac-Man currently has a fright advantage over all ghosts.
    #[must_use]
    pub fn frightened_active(&self) -> bool {
        self.ghosts.iter().any(|g| g.frightened_steps > 0)
    }

    fn ghost_step(&mut self, idx: usize) {
        let pac = (self.px, self.py);
        let Some(g) = self.ghosts.get_mut(idx) else {
            return;
        };

        if g.frightened_steps > 0 {
            g.frightened_steps -= 1;
        }

        // Candidate moves excluding reversal unless dead end.
        let dirs = [(0i32, 1i32), (0, -1), (-1, 0), (1, 0)];
        let mut candidates: Vec<(i32, i32)> = dirs
            .into_iter()
            .map(|(dx, dy)| (g.x + dx, g.y + dy))
            .filter(|(x, y)| !self.grid.is_wall(IVec2::new(*x, *y)))
            .collect();

        if candidates.is_empty() {
            return;
        }

        if g.frightened_steps > 0 {
            // flee: pick random valid
            let r = g.next_random() % candidates.len() as u64;
            if let Some(c) = candidates.get(r as usize) {
                g.x = c.0;
                g.y = c.1;
            }
            return;
        }

        // chase: minimize manhattan distance to pac-man
        candidates.sort_by_key(|(x, y)| (pac.0 - x).abs() + (pac.1 - y).abs());
        if let Some(c) = candidates.first() {
            g.x = c.0;
            g.y = c.1;
        }
    }

    /// Applies one action; returns `(reward, done)`.
    pub fn step(&mut self, action: Action) -> (f32, bool) {
        if self.done {
            return (0.0, true);
        }
        self.steps += 1;

        let (dx, dy) = action.delta();
        let nx = self.px + dx;
        let ny = self.py + dy;

        let mut reward = 0.0f32;

        if self.grid.is_wall(IVec2::new(nx, ny)) {
            reward -= 0.01;
        } else {
            self.px = nx;
            self.py = ny;
            self.facing = action_face(action);
        }

        // pellet eating
        let here = IVec2::new(self.px, self.py);
        if self.pellet_mirror.remove(&here) {
            // mirrored bookkeeping only
        }
        match self.pellets.eat(here) {
            Some(PelletKind::Pellet) => {
                reward += 1.0;
                self.score += 10;
            }
            Some(PelletKind::Power) => {
                reward += 2.0;
                self.score += 50;
                for g in &mut self.ghosts {
                    g.frightened_steps = 40; // ~40 steps of fright
                }
            }
            None => {}
        }

        // ghost steps then collision check
        for i in 0..self.ghosts.len() {
            self.ghost_step(i);

            let Some(g) = self.ghosts.get(i) else {
                continue;
            };
            if g.x == self.px && g.y == self.py {
                if g.frightened_steps > 0 {
                    reward += 5.0;
                    self.score += 200;
                    if let Some(g) = self.ghosts.get_mut(i) {
                        g.x = self.grid.ghost_spawn().x;
                        g.y = self.grid.ghost_spawn().y;
                        g.frightened_steps = 0;
                    }
                } else {
                    reward -= 10.0;
                    self.lives -= 1;
                    self.reset_positions();
                    break;
                }
            }
        }

        if self.lives == 0 || self.pellets.remaining() == 0 || self.steps >= self.max_steps {
            self.done = true;
            if self.pellets.remaining() == 0 {
                reward += 20.0;
            }
        }

        (reward, self.done)
    }

    /// Writes the observation vector:
    /// `[0..4]` wall presence up/down/left/right,
    /// `[4..8]` pellet adjacency bits,
    /// `[8..12]` normalized ghost deltas per direction,
    /// `[12]` fright active flag,
    /// `[13]` remaining pellet fraction,
    /// `[14]` current facing one-hot (4 slots at `[15..19]`),
    /// `[19]` reserved 0.
    pub fn get_observation(&self, out: &mut [f32; OBS_DIM]) {
        out.fill(0.0);

        let dirs = [(0i32, 1i32), (0, -1), (-1, 0), (1, 0)];
        for (i, (dx, dy)) in dirs.into_iter().enumerate() {
            let (nx, ny) = (self.px + dx, self.py + dy);
            out[i] = f32::from(u8::from(self.grid.is_wall(IVec2::new(nx, ny))));
            out[4 + i] =
                f32::from(u8::from(self.pellet_mirror.contains(&IVec2::new(nx, ny))));
        }

        // nearest-ghost directional pressure: sum over ghosts of sign(delta)/(1+dist)
        for g in &self.ghosts {
            let dx = g.x - self.px;
            let dy = g.y - self.py;
            let dist = (dx.abs() + dy.abs()).max(1);
            let scale = 1.0 / (dist as f32);
            let threat = if g.frightened_steps > 0 { -scale } else { scale };
            if dx.abs() >= dy.abs() {
                out[8 + usize::from(dx > 0) * 2 + usize::from(dx < 0)] += threat;
            } else {
                out[9 + usize::from(dy < 0) * 2 + usize::from(dy > 0)] += threat;
            }
        }

        out[12] = f32::from(u8::from(self.frightened_active()));
        if self.total_pellets > 0 {
            out[13] = self.pellet_mirror.len() as f32 / self.total_pellets as f32;
        }

        let face_idx = match self.facing {
            Direction::Up => 15,
            Direction::Down => 16,
            Direction::Left => 17,
            Direction::Right => 18,
        };
        out[face_idx] = 1.0;
    }

    /// Whether the episode has terminated.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.done
    }

    /// Set of remaining pellet positions (for tests/debug).
    #[must_use]
    pub fn pellet_positions(&self) -> &std::collections::HashSet<IVec2> {
        &self.pellet_mirror
    }
}

const fn action_face(action: Action) -> Direction {
    match action {
        Action::Up => Direction::Up,
        Action::Down => Direction::Down,
        Action::Left => Direction::Left,
        Action::Right => Direction::Right,
    }
}

/// Collects all pellet positions from a fresh board for fast adjacency checks.
fn initial_pellets(grid: &MazeGrid) -> std::collections::HashSet<IVec2> {
    let mut probe = PelletMap::from_maze(grid);
    let mut set = std::collections::HashSet::new();
    for y in 0..crate::maze::HEIGHT {
        for x in 0..crate::maze::WIDTH {
            let pos = IVec2::new(x as i32, y as i32);
            if grid.is_wall(pos) {
                continue;
            }
            if probe.eat(pos).is_some() {
                set.insert(pos);
            }
        }
    }
    set
}
