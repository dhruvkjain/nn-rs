use ndarray::{Array1, Array2};
pub use layer::{Layer, Initialization, Regularization};
pub use softmax::Softmax;
pub use relu::ReLu;
pub use leaky_relu::LeakyReLu;
pub use elu::ELU;
pub use selu::SELU;

pub enum LayerTypes {
    Layer(Layer),
    Softmax(Softmax),
    ReLu(ReLu),
    LeakyReLu(LeakyReLu),
    ELU(ELU),
    SELU(SELU),
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
            LayerTypes::Softmax(layer) => layer.forward(input),
            LayerTypes::ReLu(layer) => layer.forward(input),
            LayerTypes::LeakyReLu(layer) => layer.forward(input),
            LayerTypes::ELU(layer) => layer.forward(input),
            LayerTypes::SELU(layer) => layer.forward(input),
        }
    }

    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerTypes::Layer(layer) => layer.backward(grad_output),
            LayerTypes::ReLu(layer) => layer.backward(grad_output),
            LayerTypes::LeakyReLu(layer) => layer.backward(grad_output),
            LayerTypes::ELU(layer) => layer.backward(grad_output),
            LayerTypes::SELU(layer) => layer.backward(grad_output),
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
            _ => None,
        }
    }
}


pub mod layer;
pub mod relu;
pub mod softmax;
pub mod leaky_relu;
pub mod elu;
pub mod selu;