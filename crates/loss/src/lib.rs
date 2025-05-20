use ndarray::Array2;

pub trait Loss {
    fn forward(&mut self, preds: &Array2<f32>, targets: &Array2<f32>) -> f32;
    fn backward(&self, preds: &Array2<f32>) -> Array2<f32>;
}

pub mod mseloss;