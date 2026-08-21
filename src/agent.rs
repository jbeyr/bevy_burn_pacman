//! Burn 0.21 Double DQN agent for Pac-Man Lite.
//!
//! Mirrors the snake project's architecture: 3-layer MLP, circular replay
//! buffer, decoupled Double DQN Bellman targets, `.mpk` checkpointing.

use std::path::Path;

use burn::{
    module::Module,
    nn,
    optim::{AdamConfig, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor, TensorData,
    },
};
use rand::{Rng, seq::SliceRandom};

use crate::headless_env::OBS_DIM;

/// MLP Q-network: observation -> hidden -> hidden -> one Q per action.
#[derive(Module, Debug)]
pub struct PacQNetwork<B: Backend> {
    pub linear1: nn::Linear<B>,
    pub linear2: nn::Linear<B>,
    pub linear3: nn::Linear<B>,
    pub activation: nn::Relu,
}

impl<B: Backend> PacQNetwork<B> {
    /// Builds the network on `device`.
    #[must_use]
    pub fn new(device: &B::Device) -> Self {
        Self {
            linear1: nn::LinearConfig::new(OBS_DIM, 128).init(device),
            linear2: nn::LinearConfig::new(128, 128).init(device),
            linear3: nn::LinearConfig::new(128, 4).init(device),
            activation: nn::Relu::new(),
        }
    }

    /// Forward pass: `[B, OBS_DIM] -> [B, 4]`.
    #[must_use]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        self.linear3.forward(x)
    }
}

/// One experience tuple.
#[derive(Debug, Clone, Copy)]
pub struct Transition {
    pub state: [f32; OBS_DIM],
    pub action: usize,
    pub reward: f32,
    pub next_state: [f32; OBS_DIM],
    pub done: bool,
}

/// Fixed-capacity circular replay buffer.
#[derive(Debug)]
pub struct ReplayBuffer {
    buffer: Vec<Transition>,
    capacity: usize,
    position: usize,
}

impl ReplayBuffer {
    /// New empty buffer with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }

    /// Pushes a transition, overwriting the oldest when full.
    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(transition);
        } else {
            self.buffer[self.position] = transition;
            self.position = (self.position + 1) % self.capacity;
        }
    }

    /// Uniform random sample of up to `batch_size` transitions.
    #[must_use]
    pub fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<Transition> {
        let n = batch_size.min(self.buffer.len());
        self.buffer
            .choose_multiple(rng, n)
            .copied()
            .collect()
    }

    /// Number of stored transitions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether no transitions are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Double DQN agent over the [`PacQNetwork`].
pub struct DoubleDqnAgent<B: AutodiffBackend> {
    pub policy_net: PacQNetwork<B>,
    pub target_net: PacQNetwork<B>,
    pub optimizer:
        burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, PacQNetwork<B>, B>,
    device: B::Device,
    gamma: f32,
}

impl<B: AutodiffBackend> DoubleDqnAgent<B> {
    /// Creates the agent with paired policy/target networks and Adam optimizer.
    #[must_use]
    pub fn new(device: B::Device, gamma: f32) -> Self {
        Self {
            policy_net: PacQNetwork::new(&device),
            target_net: PacQNetwork::new(&device),
            optimizer: AdamConfig::new().init(),
            device,
            gamma,
        }
    }

    /// Epsilon-greedy action selection.
    pub fn select_action(
        &self,
        obs: &[f32; OBS_DIM],
        epsilon: f32,
        rng: &mut impl Rng,
    ) -> usize {
        if rng.gen_range(0.0..1.0) < epsilon {
            return rng.gen_range(0..crate::headless_env::Action::COUNT);
        }
        let q = self.q_values(obs);
        (0..4)
            .max_by(|a, b| q[*a].total_cmp(&q[*b]))
            .unwrap_or(0)
    }

    /// Raw Q-values for one observation.
    #[must_use]
    pub fn q_values(&self, obs: &[f32; OBS_DIM]) -> [f32; 4] {
        let data = TensorData::new(obs.to_vec(), [1, OBS_DIM]);
        let input: Tensor<B, 2> = Tensor::from_data(data, &self.device);
        let out = self.policy_net.forward(input);
        let data = out.into_data();
        let slice = data.as_slice::<f32>().unwrap_or(&[]);
        let mut q = [0.0f32; 4];
        for (slot, val) in q.iter_mut().zip(slice.iter()) {
            *slot = *val;
        }
        q
    }

    /// One Double DQN gradient step; returns MSE loss.
    pub fn train_step(&mut self, batch: &[Transition], lr: f64) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }
        let bs = batch.len();
        let mut states = Vec::with_capacity(bs * OBS_DIM);
        let mut next_states = Vec::with_capacity(bs * OBS_DIM);
        let mut actions = Vec::with_capacity(bs);
        let mut rewards = Vec::with_capacity(bs);
        let mut dones = Vec::with_capacity(bs);

        for t in batch {
            states.extend_from_slice(&t.state);
            next_states.extend_from_slice(&t.next_state);
            actions.push(t.action as i64);
            rewards.push(t.reward);
            dones.push(f32::from(u8::from(t.done)));
        }

        let states_t: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(states, [bs, OBS_DIM]), &self.device);
        let next_states_t: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(next_states, [bs, OBS_DIM]), &self.device);
        let actions_t: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(actions, [bs, 1]), &self.device);
        let rewards_t: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(rewards, [bs, 1]), &self.device);
        let dones_t: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(dones, [bs, 1]), &self.device);

        // Q_online(s, a)
        let q_selected = self.policy_net.forward(states_t).gather(1, actions_t);

        // a* = argmax_a Q_online(s', a)
        let best_actions: Tensor<B, 2, Int> =
            self.policy_net.forward(next_states_t.clone()).argmax(1);

        // y = r + (1-done) * gamma * Q_target(s', a*)
        let target_q = self.target_net.forward(next_states_t);
        let picked = target_q.gather(1, best_actions);
        let not_done = dones_t.neg().add_scalar(1.0);
        let targets = rewards_t
            .add(picked.mul(not_done).mul_scalar(self.gamma))
            .detach();

        let loss = q_selected.sub(targets).powf_scalar(2.0).mean();
        let value = loss
            .clone()
            .into_data()
            .as_slice::<f32>()
            .map_or(0.0, |s| s.first().copied().unwrap_or(0.0));

        let grads = loss.backward();
        let grads =
            burn::optim::GradientsParams::from_grads(grads, &self.policy_net);
        self.policy_net =
            self.optimizer.step(lr, self.policy_net.clone(), grads);

        value
    }

    /// Hard-syncs the target network from the policy network.
    pub fn sync_target(&mut self) {
        self.target_net = self.policy_net.clone();
    }

    /// Saves policy weights to `.mpk`.
    ///
    /// # Errors
    /// Returns the recorder error on I/O or serialization failure.
    pub fn save_checkpoint(&self, path: &Path) -> Result<(), burn::record::RecorderError> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder.record(self.policy_net.clone().into_record(), path.to_path_buf())
    }

    /// Loads policy weights from `.mpk` into both networks.
    ///
    /// # Errors
    /// Returns the recorder error on I/O or deserialization failure.
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<(), burn::record::RecorderError> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder.load(path.to_path_buf(), &self.device)?;
        self.policy_net = self.policy_net.clone().load_record(record);
        self.target_net = self.policy_net.clone();
        Ok(())
    }
}
