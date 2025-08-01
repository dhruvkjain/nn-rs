use ndarray::Array2;
use super::Propagate;

pub struct LeakyReLu {
    cache: Option<Array2<f32>>,
}
impl LeakyReLu {
    pub fn new() -> Self { 
        LeakyReLu { cache: None } 
    }
}
impl Propagate for LeakyReLu {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cache = Some(input.clone()); // ← Cache input instead of output
        input.mapv(|x| if x > 0.0 { x } else { 0.1 * x })
    }

    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let input = self.cache.as_ref().expect("No cache");
        let grad = input.mapv(|x| if x > 0.0 { 1.0 } else { 0.1 });
        grad_output * grad
    }
}