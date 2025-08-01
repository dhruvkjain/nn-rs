use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{rand_distr, RandomExt};
use super::Propagate;

pub struct Layer {
    weights: Array2<f32>, // shape (input_dim, output_dim)
    bias: Array1<f32>,    // shape (output_dim)
    grad_weights: Array2<f32>,
    grad_bias: Array1<f32>,
    input: Option<Array2<f32>>,
}
impl Layer {
    pub fn new(input_dim:usize, output_dim:usize) -> Self{
        let w = Array2::random((input_dim, output_dim), rand_distr::Uniform::new(-0.5, 0.5));
        let b = Array1::zeros(output_dim);
        let gw = Array2::zeros((input_dim, output_dim));
        let gb = Array1::zeros(output_dim);
        Layer { weights: w, bias: b, grad_weights: gw, grad_bias: gb, input: None }
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
