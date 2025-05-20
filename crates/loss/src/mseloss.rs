use::ndarray::Array2;
use super::Loss;

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