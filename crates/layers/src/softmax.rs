use ndarray::{Array2, Axis};
use super::Propagate;

pub struct Softmax {
    pub output: Option<Array2<f32>>,
}

impl Softmax {
    pub fn new() -> Self {
        Softmax { output: None }
    }
}

impl Propagate for Softmax {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.clone();

        // Axis(0) in 2D Array means rows
        // Axis(1) in 2D Array means cols 
        for mut row in output.axis_iter_mut(Axis(0)) {
            // For each row i.e. all classes of a label subtract max for numerical stability
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            row.mapv_inplace(|x| (x - max).exp());

            // Divide by the sum of exponentials
            let sum = row.sum();
            row.mapv_inplace(|x| x / sum);
        }

        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, _grad_output: &Array2<f32>) -> Array2<f32> {
        panic!("Softmax gradient should be handled in cross-entropy loss");
    }
}