use ndarray::{Array1, Array2};
use super::Optimizer;

pub struct NadamOptimizer {
    pub lr: f32,
    pub momentum: f32,
    pub decay_rate: f32,
    pub smoothing: f32,
    pub velocity_w: Vec<Option<Array2<f32>>>,
    pub velocity_b: Vec<Option<Array1<f32>>>,
    pub scaling_factor_w: Vec<Option<Array2<f32>>>,
    pub scaling_factor_b: Vec<Option<Array1<f32>>>,
    pub timestep: usize,
}

impl NadamOptimizer {
    pub fn new(lr: f32, momentum: f32, decay_rate: f32, smoothing: f32) -> Self {
        Self {
            lr,
            momentum,
            decay_rate,
            smoothing,
            velocity_w: Vec::new(),
            velocity_b: Vec::new(),
            scaling_factor_w: Vec::new(),
            scaling_factor_b: Vec::new(),
            timestep: 0,
        }
    }

    fn ensure_state_w(&mut self, params: &Vec<&mut Array2<f32>>) {
        if self.velocity_w.len() != params.len() {
            self.velocity_w = params.iter().map(|p| Some(Array2::zeros(p.raw_dim()))).collect();
            self.scaling_factor_w = params.iter().map(|p| Some(Array2::zeros(p.raw_dim()))).collect();
        }
    }

    fn ensure_state_b(&mut self, params: &Vec<&mut Array1<f32>>) {
        if self.velocity_b.len() != params.len() {
            self.velocity_b = params.iter().map(|p| Some(Array1::zeros(p.raw_dim()))).collect();
            self.scaling_factor_b = params.iter().map(|p| Some(Array1::zeros(p.raw_dim()))).collect();
        }
    }
}

impl Optimizer for NadamOptimizer {
    fn step_weight(&mut self, params: &mut Vec<&mut Array2<f32>>, grads: &mut Vec<&mut Array2<f32>>) {
        self.timestep += 1;
        self.ensure_state_w(params);

        let t = self.timestep as i32;

        for ((p, g), (v, s)) in params.iter_mut().zip(grads.iter())
            .zip(self.velocity_w.iter_mut().zip(self.scaling_factor_w.iter_mut())) {
            
            let vel = v.as_mut().unwrap();
            let scale = s.as_mut().unwrap();

            *vel = self.momentum * &*vel + (1.0 - self.momentum) * &**g;
            *scale = self.decay_rate * &*scale + (1.0 - self.decay_rate) * &g.mapv(|x| x * x);

            let m_hat = vel.mapv(|x| x / (1.0 - self.momentum.powi(t)));
            let s_hat = scale.mapv(|x| x / (1.0 - self.decay_rate.powi(t)));

            let nesterov = self.momentum * &m_hat + (1.0 - self.momentum) * &**g / (1.0 - self.momentum.powi(t));
            let denom = s_hat.mapv(|x| x.sqrt() + self.smoothing);

            **p -= &(self.lr * &nesterov / denom);
        }
    }

    fn step_bias(&mut self, params: &mut Vec<&mut Array1<f32>>, grads: &mut Vec<&mut Array1<f32>>) {
        self.ensure_state_b(params);

        let t = self.timestep as i32;

        for ((p, g), (v, s)) in params.iter_mut().zip(grads.iter())
            .zip(self.velocity_b.iter_mut().zip(self.scaling_factor_b.iter_mut())) {

            let vel = v.as_mut().unwrap();
            let scale = s.as_mut().unwrap();

            *vel = self.momentum * &*vel + (1.0 - self.momentum) * &**g;
            *scale = self.decay_rate * &*scale + (1.0 - self.decay_rate) * &g.mapv(|x| x * x);

            let m_hat = vel.mapv(|x| x / (1.0 - self.momentum.powi(t)));
            let s_hat = scale.mapv(|x| x / (1.0 - self.decay_rate.powi(t)));

            let nesterov = self.momentum * &m_hat + (1.0 - self.momentum) * &**g / (1.0 - self.momentum.powi(t));
            let denom = s_hat.mapv(|x| x.sqrt() + self.smoothing);

            **p -= &(self.lr * &nesterov / denom);
        }
    }
}
