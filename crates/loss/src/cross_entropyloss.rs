use ndarray::{Array1, Array2, Axis};
use layers::{Propagate, Softmax};
use super::Loss;

pub struct CrossEntropyLoss {
    pub probs: Option<Array2<f32>>,
    pub one_hot_encoded: Option<Array2<f32>>,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss {
            probs: None,
            one_hot_encoded: None,
        }
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&mut self, logits: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let probs = Softmax::new().forward(logits);
        let mut one_hot = Array2::<f32>::zeros((targets.len_of(Axis(0)), probs.len_of(Axis(1))));
        let labels: Array1<usize> = targets.iter().map(|x| *x as usize).collect();

        for (i, &class_idx) in labels.iter().enumerate() {
            one_hot[[i, class_idx]] = 1.0;
        }

        self.probs = Some(probs.clone());
        self.one_hot_encoded = Some(one_hot.clone());

        let log_probs = probs.mapv(|p| p.max(1e-9).ln());
        -(one_hot * log_probs).mean().unwrap()
    }

    fn backward(&self, _logits: &Array2<f32>) -> Array2<f32> {
        let probs = self.probs.as_ref().unwrap();
        let one_hot = self.one_hot_encoded.as_ref().unwrap();
        (probs - one_hot) / (probs.len_of(Axis(0)) as f32)
    }
}
