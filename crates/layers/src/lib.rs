use ndarray::{Array1, Array2};
pub use layer::Layer;
pub use relu::ReLu;
pub use softmax::Softmax;

pub enum LayerTypes {
    Layer(Layer),
    ReLu(ReLu),
    Softmax(Softmax)
}

pub trait Propagate {
    // returns updated values i.e. 'z'
    fn forward(&mut self, input:&Array2<f32>) -> Array2<f32>;

    // returns gradient of that layer
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32>;

    // Optional for trainable layers only
    // returns weights, gradient of weight, bias, gradient of bias
    fn params_grads(&mut self) -> Option<(&mut Array2<f32>, &mut Array2<f32>, &mut Array1<f32>, &mut Array1<f32>)> {
        None
    }
}

impl Propagate for LayerTypes {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerTypes::Layer(layer) => layer.forward(input),
            LayerTypes::ReLu(layer) => layer.forward(input),
            LayerTypes::Softmax(layer) => layer.forward(input),
        }
    }

    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerTypes::Layer(layer) => layer.backward(grad_output),
            LayerTypes::ReLu(layer) => layer.backward(grad_output),
            LayerTypes::Softmax(_) => {
                panic!("Softmax should not be used in backprop unless combined with loss")
            }
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
            LayerTypes::Softmax(_) => None,
        }
    }
}


pub mod layer;
pub mod relu;
pub mod softmax;