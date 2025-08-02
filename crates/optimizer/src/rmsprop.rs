use ndarray::{Array1, Array2};
use super::Optimizer;

pub struct RMSPropOptimizer {
    pub lr: f32,
    pub decay_rate: f32,
    pub smoothing: f32,
    pub scaling_factor_w: Vec<Option<Array2<f32>>>,
    pub scaling_factor_b: Vec<Option<Array1<f32>>>,
}

impl RMSPropOptimizer {
    pub fn new(lr: f32, decay_rate: f32, smoothing: f32) -> Self {
        RMSPropOptimizer {
            lr,
            decay_rate,
            smoothing,
            scaling_factor_w: Vec::new(),
            scaling_factor_b: Vec::new(),
        }
    }

    fn ensure_scaling_factor_w(&mut self, params: &Vec<&mut Array2<f32>>) {
        if self.scaling_factor_w.len() != params.len() {
            self.scaling_factor_w = params
                .iter()
                .map(|p| Some(Array2::zeros(p.raw_dim())))
                .collect();
        }
    }

    fn ensure_scaling_factor_b(&mut self, params: &Vec<&mut Array1<f32>>) {
        if self.scaling_factor_b.len() != params.len() {
            self.scaling_factor_b = params
                .iter()
                .map(|p| Some(Array1::zeros(p.raw_dim())))
                .collect();
        }
    }
}

impl Optimizer for RMSPropOptimizer {
    fn step_weight(&mut self, params: &mut Vec<&mut Array2<f32>>, grads: &mut Vec<&mut Array2<f32>>) {
        self.ensure_scaling_factor_w(params);

        for ((p, g), s) in params.iter_mut().zip(grads.iter()).zip(self.scaling_factor_w.iter_mut()) {
            let scale = s.as_mut().unwrap();
            *scale = (self.decay_rate * &*scale) + (1.0 - self.decay_rate) * &g.mapv(|x| x * x);
            let adjusted = g.mapv(|x| x) / (&*scale + self.smoothing).mapv(|x| x.sqrt());
            **p -= &(self.lr * &adjusted);
        }
    }

    fn step_bias(&mut self, params: &mut Vec<&mut Array1<f32>>, grads: &mut Vec<&mut Array1<f32>>) {
        self.ensure_scaling_factor_b(params);

        for ((p, g), s) in params.iter_mut().zip(grads.iter()).zip(self.scaling_factor_b.iter_mut()) {
            let scale = s.as_mut().unwrap();
            *scale = self.decay_rate * &*scale + (1.0 - self.decay_rate) * &g.mapv(|x| x * x);
            let adjusted = g.mapv(|x| x) / (&*scale + self.smoothing).mapv(|x| x.sqrt());
            **p -= &(self.lr * &adjusted);
        }
    }
}