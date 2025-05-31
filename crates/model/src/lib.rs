use loss::Loss;
use optimizer::Optimizer;
use layers::{LayerTypes, Propagate};
use ndarray::Array2;


pub struct NN<S: Loss, O: Optimizer> {
    pub layers: Vec<LayerTypes>,
    pub loss_fn: S,
    pub optim: O,
}

impl<S: Loss, O: Optimizer> NN<S, O> {
    /// Forward pass through all layers.
    pub fn forward_all(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let mut z = x.clone();
        for layer in  self.layers.iter_mut(){
            z = layer.forward(&z);
        }
        z
    }

    /// Backward pass: loss gradient → all layers (in reverse)
    pub fn backward_all(&mut self, grad_loss: &Array2<f32>) {
        let mut grad = grad_loss.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    /// Single training step on batch (inputs, targets)
    pub fn train_step(&mut self, x: &Array2<f32>, y: &Array2<f32>) -> f32 {
        // Forward
        let preds = self.forward_all(x);
        // Loss
        let loss = self.loss_fn.forward(&preds, y);
        let grad_loss = self.loss_fn.backward(&preds);
        // Backward
        self.backward_all(&grad_loss);
        // Collect all params and grads
        let mut all_weights = Vec::new();
        let mut all_grad_weights = Vec::new();
        let mut all_bias = Vec::new();
        let mut all_grad_bias = Vec::new();
        for layer in self.layers.iter_mut() {
            match layer.params_grads() {
                Some((weights ,grad_weights, bias, grad_bias)) => {
                    all_weights.push(weights);
                    all_grad_weights.push(grad_weights);
                    all_bias.push(bias);
                    all_grad_bias.push(grad_bias);
                }
                None => { 
                    // println!("Activation Layer weights and params can't be collected") 
                }
            }
        }
        // Update
        self.optim.step_weight(&mut all_weights, &mut all_grad_weights);
        self.optim.step_bias(&mut all_bias, &mut all_grad_bias);
        loss
    }
}
