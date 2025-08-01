use ndarray::{Array1, Array2};
use super::Optimizer;

pub struct MomentumOptimizer {
    pub lr: f32,
    pub momentum: f32,
    pub velocity_w: Vec<Option<Array2<f32>>>,
    pub velocity_b: Vec<Option<Array1<f32>>>,
}

impl MomentumOptimizer {
    pub fn new(lr: f32, momentum: f32) -> Self {
        MomentumOptimizer {
            lr,
            momentum,
            velocity_w: Vec::new(),
            velocity_b: Vec::new(),
        }
    }

    fn ensure_velocity_w(&mut self, params: &Vec<&mut Array2<f32>>) {
        if self.velocity_w.len() != params.len() {
            self.velocity_w = params
                .iter()
                .map(|p| Some(Array2::zeros(p.raw_dim())))
                .collect();
        }
    }

    fn ensure_velocity_b(&mut self, params: &Vec<&mut Array1<f32>>) {
        if self.velocity_b.len() != params.len() {
            self.velocity_b = params
                .iter()
                .map(|p| Some(Array1::zeros(p.raw_dim())))
                .collect();
        }
    }
}

impl Optimizer for MomentumOptimizer {
    fn step_weight(&mut self, params: &mut Vec<&mut Array2<f32>>, grads: &mut Vec<&mut Array2<f32>>) {
        self.ensure_velocity_w(params);

        for i in 0..params.len() {
            let v = self.velocity_w[i].as_mut().unwrap();
            *v = self.momentum * &*v - self.lr * &*grads[i];
            *params[i] += &*v;
        }
    }

    fn step_bias(&mut self, params: &mut Vec<&mut Array1<f32>>, grads: &mut Vec<&mut Array1<f32>>) {
        self.ensure_velocity_b(params);

        for i in 0..params.len() {
            let v = self.velocity_b[i].as_mut().unwrap();
            *v = self.momentum * &*v - self.lr * &*grads[i];
            *params[i] += &*v;
        }
    }
}
