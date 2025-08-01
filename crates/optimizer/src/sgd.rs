use ndarray::{Array1, Array2};
use super::Optimizer;

pub struct SGDOptimizer {
    pub lr: f32,
}
impl Optimizer for SGDOptimizer {
    fn step_weight(&mut self, params: &mut Vec<&mut Array2<f32>>, grads: &mut Vec<&mut Array2<f32>>) {
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            **p -= &(self.lr * &(**g));
        }
    }
    fn step_bias(&mut self, params: &mut Vec<&mut Array1<f32>>, grads: &mut Vec<&mut Array1<f32>>) {
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            **p -= &(self.lr * &(**g));
        }
    }
}