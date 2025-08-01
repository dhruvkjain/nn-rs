use ndarray::{Array1, Array2};
pub use sgd::SGDOptimizer;
pub use momentum::MomentumOptimizer;

pub trait Optimizer {
    fn step_weight(&mut self, params: &mut Vec<&mut Array2<f32>>, grads: &mut Vec<&mut Array2<f32>>);
    fn step_bias(&mut self, params: &mut Vec<&mut Array1<f32>>, grads: &mut Vec<&mut Array1<f32>>);
}

pub mod sgd;
pub mod momentum;