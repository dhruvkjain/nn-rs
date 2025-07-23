use ndarray::{Array1, Array2, Axis};
use layers::{Propagate, Softmax};
use super::Loss;

pub struct CrossEntropyLoss {
    pub targets: Option<Array2<f32>>,
    pub probs: Option<Array2<f32>>,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss {
            targets: None,
            probs: None,
        }
    }
}
impl Loss for CrossEntropyLoss{ 
    fn forward(&mut self, preds: &Array2<f32>, labels: &Array2<f32>) -> f32 {
        self.targets = Some(labels.clone());

        // pass through softmax and store the probabilities
        let mut softmax_layer = Softmax::new();
        let probs = softmax_layer.forward(preds);
        self.probs = Some(probs.clone());

        // Axis(0) in 2D Array means rows
        // Axis(1) in 2D Array means cols
        let pred_classes: Array1<usize> = probs
            .axis_iter(Axis(0))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        let true_classes: Array1<usize> = labels.iter().map(|x| *x as usize).collect();

        let correct = pred_classes
            .iter()
            .zip(true_classes.iter())
            .filter(|(p, t)| p == t)
            .count();

        (1.0 - (correct as f32 / labels.len() as f32))*100.0
    }

    fn backward(&self, _preds: &Array2<f32>) -> Array2<f32> {
        let targets = self.targets.as_ref().expect("No targets recorded");
        let probabilities = self.probs.as_ref().expect("No probs recorded");
        
        let labels: Array1<usize> = targets.iter().map(|x| *x as usize).collect();

        let mut one_hot = Array2::<f32>::zeros((targets.len_of(Axis(0)), probabilities.len_of(Axis(1))));

        for (i, &class_idx) in labels.iter().enumerate() {
            one_hot[[i, class_idx]] = 1.0;
        }

        // Gradient: dL/dz = (probabilities of predictions - target)
        (probabilities - &one_hot) / targets.len() as f32
    }
}