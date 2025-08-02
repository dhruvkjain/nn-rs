use ndarray::Array2;
use super::Propagate;

pub struct LeakyReLu {
    alpha: f32,
    cache: Option<Array2<f32>>,
}
impl LeakyReLu {
    pub fn new(alpha: f32) -> Self { 
        LeakyReLu { 
            alpha,
            cache: None 
        } 
    }
}
impl Propagate for LeakyReLu {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cache = Some(input.clone());
        input.mapv(|x| if x > 0.0 { x } else { self.alpha * x })
    }

    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let input = self.cache.as_ref().expect("No cache");
        let grad = input.mapv(|x| if x > 0.0 { 1.0 } else { self.alpha });
        grad_output * grad
    }
}