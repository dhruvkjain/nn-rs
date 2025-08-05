use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{rand_distr, RandomExt};
use super::Propagate;

pub enum Initialization {
    He,
    Glorot,
    LeCun
}

pub enum Regularization {
    None,
    L1{lambda: f32},
    L2{lambda: f32},
    ElasticNet { l1: f32, l2: f32 },
}

pub struct Layer {
    weights: Array2<f32>, // shape (input_dim, output_dim)
    bias: Array1<f32>,    // shape (output_dim)
    grad_weights: Array2<f32>,
    grad_bias: Array1<f32>,
    input: Option<Array2<f32>>,
    regularization: Regularization,
}


impl Layer {
    pub fn new(input_dim:usize, output_dim:usize, distribution: Initialization, regularization: Regularization) -> Self{
        let stddev = match distribution {
            Initialization::He => (2.0 / input_dim as f32).sqrt(),
            Initialization::Glorot => (1.0 / (input_dim as f32 + output_dim as f32)).sqrt(),
            Initialization::LeCun => (1.0 / (input_dim as f32)).sqrt(),
        };
        let distr = rand_distr::Normal::new(0.0, stddev).expect("Stddev for initialization must be positive");
        let w = Array2::random((input_dim, output_dim), distr);
        let b = Array1::zeros(output_dim);
        let gw = Array2::zeros((input_dim, output_dim));
        let gb = Array1::zeros(output_dim);
        Layer { 
            weights: w, 
            bias: b, 
            grad_weights: gw, 
            grad_bias: gb, 
            input: None, 
            regularization
        }
    }

    pub fn set_params(&mut self, weights: Array2<f32>, bias: Array1<f32>) {
        self.weights = weights;
        self.bias = bias;
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
        let mut grad_w = input.t().dot(grad_output);
        let grad_b = grad_output.sum_axis(Axis(0));

        match self.regularization {
            Regularization::L1{lambda} => {
                grad_w += &(self.weights.mapv(|x| x.signum()) * lambda);
            }
            Regularization::L2{lambda} => {
                grad_w += &(self.weights.mapv(|w| 2.0 * lambda * w));
            }
            Regularization::ElasticNet { l1, l2 } => {
                let l1_term = self.weights.mapv(|x| x.signum()) * l1;
                let l2_term = self.weights.mapv(|w| 2.0 * l2 * w);
                grad_w += &(l1_term + l2_term);
            }
            Regularization::None => {}
        }
        
        self.grad_weights = grad_w;
        self.grad_bias = grad_b;
        // Propagate gradient to inputs
        grad_output.dot(&self.weights.t())
    }

    fn params_grads(&mut self) -> Option<(&mut Array2<f32>, &mut Array2<f32>, &mut Array1<f32>, &mut Array1<f32>)> {
        Some((&mut self.weights, &mut self.grad_weights, &mut self.bias ,&mut self.grad_bias))
    }
}
