use ndarray::{Array1, Array2, Axis};
use layers::{Propagate, Softmax};
use super::Loss;

pub struct MSELoss{
    pub probs: Option<Array2<f32>>,
    pub one_hot_encoded: Option<Array2<f32>>,
}
impl MSELoss {
    pub fn new() -> Self {
        MSELoss { probs: None, one_hot_encoded: None }
    }
}
impl Loss for MSELoss{ 
    fn forward(&mut self, preds: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let mut softmax_layer = Softmax::new();
        let probs = softmax_layer.forward(preds);

        let mut one_hot = Array2::<f32>::zeros((targets.len_of(Axis(0)), probs.len_of(Axis(1))));
        let labels: Array1<usize> = targets.iter().map(|x| *x as usize).collect();

        for (i, &class_idx) in labels.iter().enumerate() {
            one_hot[[i, class_idx]] = 1.0;
        }

        self.one_hot_encoded = Some(one_hot.clone());
        self.probs = Some(probs.clone());
        let diff = probs - one_hot;
        diff.mapv(|x| x.powi(2)).mean().unwrap()
    }
    fn backward(&self, preds: &Array2<f32>) -> Array2<f32> {
        let probabilities = self.probs.as_ref().expect("No cached probabilities");
        let one_hot = self.one_hot_encoded.as_ref().expect("No cached targets");
        (2.0 / (preds.len() as f32)) * (probabilities - one_hot)
    }
}