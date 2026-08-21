//! Score, lives, HUD, and game-over overlays.
//!
//! Owns the `Score`/`Lives` resources and all UI text. Reacts to gameplay
//! events via observers; overlays toggle visibility from the global
//! [`crate::GameState`].

#![allow(dead_code)]
#![allow(clippy::needless_pass_by_value)]

use bevy::prelude::*;
use bevy::text::FontSize;

use crate::events::{GhostCollisionEvent, LevelClearedEvent, PelletEatenEvent, PowerPelletEatenEvent};
use crate::ghosts::{Ghost, GhostMode};
use crate::maze::PelletMap;
use crate::GameState;

// LifeLostEvent fires from main once restart flow is complete.
/// Player's current score.
#[derive(Debug, Resource, Default)]
pub struct Score(pub u32);

/// Remaining extra lives. Starts at three per session.
#[derive(Debug, Resource)]
pub struct Lives(pub u32);

impl Default for Lives {
    fn default() -> Self {
        Self(3)
    }
}

/// Best score this session.
#[derive(Debug, Resource, Default)]
pub struct HighScore(pub u32);

/// Marker for the score display.
#[derive(Component, Debug)]
pub struct ScoreText;

/// Marker for the lives display.
#[derive(Component, Debug)]
pub struct LivesText;

/// Marker for the game-over overlay root.
#[derive(Component, Debug)]
pub struct GameOverOverlay;

/// Marker for the level-cleared overlay root.
#[derive(Component, Debug)]
pub struct WinOverlay;

/// Plugin wiring score/lives resources, HUD, and observers.
#[derive(Debug)]
pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Score>()
            .init_resource::<Lives>()
            .init_resource::<HighScore>()
            .add_systems(Startup, setup_hud)
            .add_systems(
                Update,
                (update_hud_text, update_overlays).run_if(in_state(GameState::Playing)),
            )
            .add_observer(on_pellet_eaten)
            .add_observer(on_power_pellet)
            .add_observer(on_ghost_collision);
    }
}

fn setup_hud(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(8.0),
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::SpaceBetween,
                padding: UiRect::new(Val::Px(16.0), Val::Px(16.0), Val::Px(0.0), Val::Px(0.0)),
                ..default()
            },
        ))
        .with_children(|bar| {
            bar.spawn((
                Text::new("Score: 0"),
                TextFont { font_size: FontSize::Px(24.0), ..default() },
                TextColor(Color::WHITE),
                ScoreText,
            ));
            bar.spawn((
                Text::new("Lives: 3"),
                TextFont { font_size: FontSize::Px(24.0), ..default() },
                TextColor(Color::WHITE),
                LivesText,
            ));
        });

    // Centered game-over overlay (hidden until needed).
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        Visibility::Hidden,
        GameOverOverlay,
    ))
    .with_children(|overlay| {
        overlay.spawn((
            Text::new("GAME OVER - press R to restart"),
            TextFont { font_size: FontSize::Px(40.0), ..default() },
            TextColor(Color::srgb(1.0, 0.3, 0.3)),
        ));
    });

    // Centered win overlay.
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        Visibility::Hidden,
        WinOverlay,
    ))
    .with_children(|overlay| {
        overlay.spawn((
            Text::new("LEVEL CLEARED!"),
            TextFont { font_size: FontSize::Px(40.0), ..default() },
            TextColor(Color::srgb(0.4, 1.0, 0.5)),
        ));
    });
}

fn update_hud_text(
    score: Res<Score>,
    lives: Res<Lives>,
    mut texts: ParamSet<(
        Query<&mut Text, With<ScoreText>>,
        Query<&mut Text, With<LivesText>>,
    )>,
) {
    if score.is_changed() {
        let value = score.0;
        for mut text in &mut texts.p0() {
            **text = format!("Score: {value}");
        }
    }
    if lives.is_changed() {
        let value = lives.0;
        for mut text in &mut texts.p1() {
            **text = format!("Lives: {value}");
        }
    }
}

fn update_overlays(
    game_state: Res<State<GameState>>,
    mut overlays: ParamSet<(
        Query<&mut Visibility, With<GameOverOverlay>>,
        Query<&mut Visibility, With<WinOverlay>>,
    )>,
) {
    let want_game_over = *game_state.get() == GameState::GameOver;
    let want_win = *game_state.get() == GameState::Win;
    for mut vis in &mut overlays.p0() {
        *vis = if want_game_over { Visibility::Visible } else { Visibility::Hidden };
    }
    for mut vis in &mut overlays.p1() {
        *vis = if want_win { Visibility::Visible } else { Visibility::Hidden };
    }
}

fn on_pellet_eaten(
    _trigger: On<PelletEatenEvent>,
    mut score: ResMut<Score>,
    pellets: Res<PelletMap>,
    mut commands: Commands,
) {
    score.0 += 10;
    if pellets.remaining() == 0 {
        commands.trigger(LevelClearedEvent);
    }
}

fn on_power_pellet(_trigger: On<PowerPelletEatenEvent>, mut score: ResMut<Score>) {
    score.0 += 50;
}

fn on_ghost_collision(
    trigger: On<GhostCollisionEvent>,
    ghost_q: Query<&Ghost>,
    mut lives: ResMut<Lives>,
) {
    let ghost_entity = trigger.event().ghost;
    let Ok(ghost) = ghost_q.get(ghost_entity) else {
        return;
    };
    if matches!(ghost.mode, GhostMode::Frightened { .. }) {
        return; // eating ghosts is a bonus, not a death
    }
    if lives.0 > 0 {
        lives.0 -= 1;
    }
}
