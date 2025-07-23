use ndarray::Array2;
pub use mseloss::MSELoss;
pub use cross_entropyloss::CrossEntropyLoss;

pub trait Loss {
    // returns loss
    fn forward(&mut self, preds: &Array2<f32>, targets: &Array2<f32>) -> f32;
    
    // returns gradient
    fn backward(&self, preds: &Array2<f32>) -> Array2<f32>;
}

pub mod mseloss;
pub mod cross_entropyloss;