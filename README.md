# Bevy + Burn: Pac-Man Lite (`bevy_burn_pacman`)
### **Project 04 of the Pure Rust Machine Learning & Game AI Curriculum**

Pellet-foraging maze with scripted ghost adversaries. Playable Bevy 0.19 game first, then a Double DQN agent (Burn 0.21) trained headlessly against the same ghost AI.

---

## Key Features

1. **Playable Game First:** Grid maze, pellets, power pellets, 4 ghosts with chase/frightened/eaten state machine, lives, score HUD.
2. **Scripted Adversaries:** Ghosts use classic direction-choice heuristics (target Pac-Man, flee when frightened) — real in-world pressure, no reward shaping required.
3. **DQN Agent:** Relative wall/pellet/ghost observation vector; headless training against the identical ghost logic.
4. **Dual Execution:** `--headless --episodes N` for fast policy training; default run for 60 FPS visual play.

## Quickstart

```bash
cargo run --release                          # play the game
cargo run --release -- --headless --episodes 2000   # train the agent
cargo test                                   # unit tests
```
