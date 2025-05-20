use ndarray::{Array1, Array2};

pub trait Propagate {
    fn forward(&mut self, input:&Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32>;

    // Optional for trainable layers only
    fn params_grads(&mut self) -> Option<(&mut Array2<f32>, &mut Array2<f32>, &mut Array1<f32>, &mut Array1<f32>)> {
        None
    }
}

pub mod layer;
pub mod relu;