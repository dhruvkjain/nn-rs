use ndarray::Array2;
use super::Propagate;

pub struct SELU {
    alpha: f32,
    scale: f32,
    cache: Option<Array2<f32>>,
}
impl SELU {
    pub fn new(alpha: f32, scale: f32) -> Self {
        SELU { 
            alpha,
            scale,
            cache: None 
        }
    }
}
impl Propagate for SELU {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cache = Some(input.clone());

        input.mapv(|x| { self.scale * (if x > 0.0 { x } else { self.alpha * (x.exp() - 1.0) }) })
    }

    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let input = self.cache.as_ref().expect("No cache");

        let grad = input.mapv(|x| { self.scale * if x > 0.0 { 1.0 } else { self.alpha * x.exp() } });
        grad_output * grad
    }
}
