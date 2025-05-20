use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{rand_distr, RandomExt};


pub trait Propagate {
    fn forward(&mut self, input:&Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32>;

    // Optional for trainable layers only
    fn params_grads(&mut self) -> Option<(&mut Array2<f32>, &mut Array2<f32>, &mut Array1<f32>, &mut Array1<f32>)> {
        None
    }
}

pub struct Layer {
    weights: Array2<f32>, // shape (input_dim, output_dim)
    bias: Array1<f32>,    // shape (output_dim)
    grad_weights: Array2<f32>,
    grad_bias: Array1<f32>,
    input: Option<Array2<f32>>,
}
impl Layer {
    pub fn new(input_dim:usize, output_dim:usize) -> Self{
        let w = Array2::random((input_dim, output_dim), rand_distr::StandardNormal);
        let b = Array1::zeros(output_dim);
        let gw = Array2::zeros((input_dim, output_dim));
        let gb = Array1::zeros(output_dim);
        Layer { weights: w, bias: b, grad_weights: gw, grad_bias: gb, input: None }
    }
}
impl Propagate for Layer {
    fn forward(&mut self, input:&Array2<f32>) -> Array2<f32> {
        self.input = Some(input.clone());
        input.dot(&self.weights) + &self.bias
    }

    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let input = self.input.as_ref().expect("No cache");
        // Compute gradients
        let grad_w = input.t().dot(grad_output);
        let grad_b = grad_output.sum_axis(Axis(0));

        self.grad_weights = grad_w;
        self.grad_bias = grad_b;
        // Propagate gradient to inputs
        grad_output.dot(&self.weights.t())
    }

    fn params_grads(&mut self) -> Option<(&mut Array2<f32>, &mut Array2<f32>, &mut Array1<f32>, &mut Array1<f32>)> {
        Some((&mut self.weights, &mut self.grad_weights, &mut self.bias ,&mut self.grad_bias))
    }
}



pub trait Loss {
    fn forward(&mut self, preds: &Array2<f32>, targets: &Array2<f32>) -> f32;
    fn backward(&self, preds: &Array2<f32>) -> Array2<f32>;
}

pub struct MSELoss{
    pub targets: Option<Array2<f32>>
}
impl MSELoss {
    pub fn new() -> Self {
        MSELoss { targets: None }
    }
}
impl Loss for MSELoss{ 
    fn forward(&mut self, preds: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        self.targets = Some(targets.clone());
        let diff = preds - targets;
        diff.mapv(|x| x.powi(2)).mean().unwrap()
    }
    fn backward(&self, preds: &Array2<f32>) -> Array2<f32> {
        let targets = self.targets.as_ref().expect("No cached targets");
        (2.0 / (preds.len() as f32)) * (preds - targets)
    }
}



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



pub trait Optimizer {
    fn step_weight(&self, params: &mut Vec<&mut Array2<f32>>, grads: &mut Vec<&mut Array2<f32>>);
    fn step_bias(&self, params: &mut Vec<&mut Array1<f32>>, grads: &mut Vec<&mut Array1<f32>>);
}

pub struct SGDOptimizer {
    pub lr: f32,
}
impl Optimizer for SGDOptimizer {
    fn step_weight(&self, params: &mut Vec<&mut Array2<f32>>, grads: &mut Vec<&mut Array2<f32>>) {
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            **p -= &(self.lr * &(**g));
        }
    }
    fn step_bias(&self, params: &mut Vec<&mut Array1<f32>>, grads: &mut Vec<&mut Array1<f32>>) {
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            **p -= &(self.lr * &(**g));
        }
    }
}