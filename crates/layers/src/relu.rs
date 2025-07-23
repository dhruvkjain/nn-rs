use ndarray::Array2;
use super::Propagate;

pub struct ReLu {
    cache: Option<Array2<f32>>,
}
impl ReLu {
    pub fn new() -> Self { 
        ReLu { cache: None } 
    }
}
impl Propagate for ReLu {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let mask = input.mapv(|x| x.max(0.0));
        self.cache = Some(mask.clone());
        mask
    }
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let mask = self.cache.as_ref().expect("No cache");
        grad_output * mask
    }
}