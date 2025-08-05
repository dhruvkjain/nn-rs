use ndarray::{Array1, Array2, Axis};

pub use layers::*;
pub use loss::*;
pub use optimizer::*;
pub use savemodel::*;
pub use loadmodel::*;

pub struct NN<S: Loss, O: Optimizer> {
    pub layers: Vec<LayerTypes>,
    pub loss_fn: S,
    pub optim: O,
    pub regularization: Regularization,
}

impl<S: Loss, O: Optimizer> NN<S, O> {
    pub fn compute_confusion_matrix(y_true: &Array1<usize>, y_pred: &Array1<usize>, num_classes: usize) -> Array2<usize> {
        let mut cm = Array2::<usize>::zeros((num_classes, num_classes));
        for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
            cm[[t, p]] += 1;
        }
        cm
    }

    pub fn compute_metrics(cm: &Array2<usize>) {
        let num_classes = cm.nrows();

        for class in 0..num_classes {
            let tpos = cm[[class, class]];
            let fpos = cm.column(class).sum() - tpos;
            let fneg = cm.row(class).sum() - tpos;
            let tneg = cm.sum() - (tpos + fpos + fneg);
            let precision: f32 = tpos as f32 / (tpos + fpos) as f32;
            let recall: f32 = tpos as f32 / (tpos + fneg) as f32;

            println!("Class {}: TP = {}, FP = {}, FN = {}, TN = {}", class, tpos, fpos, fneg, tneg);
            println!("Precision = {}", precision);
            println!("Recall(Sensitivity) = {}", recall);
            println!("F1-score: = {}",  2.0 * (precision * recall) / (precision + recall));
            println!("---------------------------------------------");
        }
    }

    // Forward pass through all layers.
    pub fn forward_all(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let mut z = x.clone();
        for layer in  self.layers.iter_mut(){
            z = layer.forward(&z);
        }
        z
    }

    // Backward pass: loss gradient → all layers (in reverse)
    pub fn backward_all(&mut self, grad_loss: &Array2<f32>) {
        let mut grad = grad_loss.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    // Single training step on batch (inputs, targets)
    pub fn train_step(&mut self, x: &Array2<f32>, y: &Array2<f32>, itertation: usize, save_at: usize, save_path: &str) -> (f32, f32) {
        // Forward
        let preds = self.forward_all(x);
        // Probabilities
        let probs = Softmax::new().forward(&preds);
        // --------------------Prediction Labels--------------------
        let pred_labels: Array1<usize> = probs
            .axis_iter(Axis(0))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        let target_labels: Array1<usize> = y.iter().map(|x| *x as usize).collect();

        // --------------------Accuracy--------------------
        let correct = pred_labels
            .iter()
            .zip(target_labels.iter())
            .filter(|(p, t)| p == t)
            .count();

        let accuracy = (correct as f32 / y.len() as f32) * 100.0;
        // Loss
        let loss = self.loss_fn.forward(&preds, y);

        // Add regularization penalty
        let mut reg_penalty = 0.0;
        for layer in self.layers.iter_mut() {
            if let Some((weights, _, _, _)) = layer.params_grads() {
                match self.regularization {
                    Regularization::L1{lambda} => {
                        reg_penalty += lambda * weights.mapv(|x| x.abs()).sum();
                    }
                    Regularization::L2{lambda} => {
                        reg_penalty += lambda * weights.mapv(|w| w * w).sum();
                    }
                    Regularization::ElasticNet { l1, l2 } => {
                        let l1_term = l1 * weights.mapv(f32::abs).sum();
                        let l2_term = l2 * weights.mapv(|w| w * w).sum();
                        reg_penalty += l1_term + l2_term;
                    }
                    Regularization::None => {}
                }
            }
        }

        let final_loss = loss + reg_penalty;

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

        // save the last weights and bias
        if itertation == save_at {
            if let Err(e) = save_model(all_weights, all_bias, save_path) {
                eprintln!("Failed to save weights: {}", e);
            }
        }

        (final_loss, accuracy)
    }

    pub fn test_step(&mut self, x: &Array2<f32>, y: &Array2<f32>) -> (f32, f32) {
        let preds = self.forward_all(x);
        let probs = Softmax::new().forward(&preds);
        let loss = self.loss_fn.forward(&preds, y);
        
        // --------------------Prediction Labels--------------------
        let pred_labels: Array1<usize> = probs
            .axis_iter(Axis(0))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        let target_labels: Array1<usize> = y.iter().map(|x| *x as usize).collect();

        // --------------------Accuracy--------------------
        let correct = pred_labels
            .iter()
            .zip(target_labels.iter())
            .filter(|(p, t)| p == t)
            .count();

        let accuracy = (correct as f32 / y.len() as f32) * 100.0;

        // --------------------Confusion Matrix & Metrics--------------------
        let num_classes = probs.shape()[1];
        let cm = Self::compute_confusion_matrix(&target_labels, &pred_labels, num_classes);
        println!("Confusion Matrix:\n{cm}");
        Self::compute_metrics(&cm);

        (loss, accuracy)
    }
}

pub mod savemodel;
pub mod loadmodel;