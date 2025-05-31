use layer::Layer;
use ndarray::{Array1, Array2};
use relu::ReLu;

pub enum LayerTypes {
    Layer(Layer),
    ReLu(ReLu)
}

pub trait Propagate {
    fn forward(&mut self, input:&Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32>;

    // Optional for trainable layers only
    fn params_grads(&mut self) -> Option<(&mut Array2<f32>, &mut Array2<f32>, &mut Array1<f32>, &mut Array1<f32>)> {
        None
    }
}

impl Propagate for LayerTypes {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerTypes::Layer(layer) => layer.forward(input),
            LayerTypes::ReLu(layer) => layer.forward(input),
        }
    }

    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerTypes::Layer(layer) => layer.backward(grad_output),
            LayerTypes::ReLu(layer) => layer.backward(grad_output),
        }
    }

    fn params_grads(&mut self) -> Option<(
        &mut Array2<f32>,
        &mut Array2<f32>,
        &mut Array1<f32>,
        &mut Array1<f32>,
    )> {
        match self {
            LayerTypes::Layer(layer) => layer.params_grads(),
            LayerTypes::ReLu(_) => None,
        }
    }
}


pub mod layer;
pub mod relu;