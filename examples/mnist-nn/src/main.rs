use polars::prelude::*;
use std::error::Error;
use ndarray::Array2;

use model::{layer::Regularization, *};

pub fn load_training_data() -> Result<(Array2<f32>, Array2<f32>), Box<dyn Error>> {
    let train_lf = LazyCsvReader::new("../test_data/mnist/mnist_train.csv")
    .with_has_header(true)
    .finish()?;

    // let df = q.clone().with_streaming(true).collect()?;
    let training_labels = train_lf
        .clone()
        .with_streaming(true)
        .select([col("label")])
        .collect()?;

    let training_data = train_lf
        .clone()
        .with_streaming(true)
        .drop([col("label")])
        .collect()?;


    // ----------------------------
    // to_ndarray docs of polars:
    // https://docs.rs/polars/0.47.0/polarsArrayBase<OwnedRepr<f32/prelude/struct.DataFrame.html#method.to_ndarray
    // ----------------------------
    let mut traning_data_ndarray = training_data
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap();
    let mut training_labels_ndarray = training_labels
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap();

    // dataset values were from 0 - 255
    traning_data_ndarray = traning_data_ndarray / 255.0;
    training_labels_ndarray = training_labels_ndarray;

    let data_dimensions:&[usize] = traning_data_ndarray.shape();
    let labels_dimensions:&[usize] = training_labels_ndarray.shape();

    // println!("{}", traning_data_ndarray);
    // println!("{}", training_labels_ndarray);
    println!("Training dataset dimensions");
    println!("DATA: {}, {}", data_dimensions[0], data_dimensions[1]);
    println!("LABELS: {}, {}", labels_dimensions[0], labels_dimensions[1]);
    Ok((traning_data_ndarray, training_labels_ndarray))
}

pub fn load_testing_data() -> Result<(Array2<f32>, Array2<f32>), Box<dyn Error>> {
    let test_lf = LazyCsvReader::new("../test_data/mnist/mnist_test.csv")
    .with_has_header(true)
    .finish()?;

    // let df = q.clone().with_streaming(true).collect()?;
    let testing_labels = test_lf
        .clone()
        .with_streaming(true)
        .select([col("label")])
        .collect()?;

    let testing_data = test_lf
        .clone()
        .with_streaming(true)
        .drop([col("label")])
        .collect()?;


    // ----------------------------
    // to_ndarray docs of polars:
    // https://docs.rs/polars/0.47.0/polarsArrayBase<OwnedRepr<f32/prelude/struct.DataFrame.html#method.to_ndarray
    // ----------------------------
    let mut testing_data_ndarray = testing_data
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap();
    let mut testing_labels_ndarray = testing_labels
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap();

    // dataset values were from 0 - 255
    testing_data_ndarray = testing_data_ndarray / 255.0;
    testing_labels_ndarray = testing_labels_ndarray;

    let data_dimensions:&[usize] = testing_data_ndarray.shape();
    let labels_dimensions:&[usize] = testing_labels_ndarray.shape();

    // println!("{}", traning_data_ndarray);
    // println!("{}", training_labels_ndarray);
    println!("Testing dataset dimensions");
    println!("DATA: {}, {}", data_dimensions[0], data_dimensions[1]);
    println!("LABELS: {}, {}", labels_dimensions[0], labels_dimensions[1]);
    Ok((testing_data_ndarray, testing_labels_ndarray))
}


fn main() -> Result<(), Box<dyn Error>> {

    // here x is of [batch size, number of classes]
    let (x, y) = load_training_data()?;

    // ----------------------MODEL-----------------------
    let mut nn = NN {
        layers: vec![
            LayerTypes::Layer(Layer::new(784, 64, Initialization::He, Regularization::ElasticNet { l1: 0.0001, l2: 0.0001 })),
            LayerTypes::ELU(ELU::new(1.0)),
            LayerTypes::Layer(Layer::new(64, 32, Initialization::He, Regularization::ElasticNet { l1: 0.0001, l2: 0.0001 })),
            LayerTypes::ELU(ELU::new(1.0)),
            LayerTypes::Layer(Layer::new(32, 16, Initialization::He, Regularization::ElasticNet { l1: 0.0001, l2: 0.0001 })),
            LayerTypes::ELU(ELU::new(1.0)),
            LayerTypes::Layer(Layer::new(16, 10, Initialization::He, Regularization::ElasticNet { l1: 0.0001, l2: 0.0001 })),
        ],
        loss_fn: CrossEntropyLoss { probs: None, one_hot_encoded:None },
        optim: NadamOptimizer { 
            lr: 0.0005, 
            momentum: 0.9,
            decay_rate: 0.999, 
            smoothing: 1e-7 as f32,
            velocity_w: Vec::new(),
            velocity_b: Vec::new(),
            scaling_factor_w: Vec::new(), 
            scaling_factor_b: Vec::new(),
            timestep: 0,
        },
        regularization: Regularization::ElasticNet { l1: 0.0001, l2: 0.0001 },
    };
    
    // let num_layers = nn.layers
    //     .iter()
    //     .filter(|l| matches!(l, LayerTypes::Layer(_)))
    //     .count();
    // let (loaded_weights, loaded_bias) = load_model_weights_and_biases("../test_data/mnist", num_layers)?;

    // for (i, layer) in nn.layers.iter_mut().filter_map(|l| match l {
    //     LayerTypes::Layer(layer) => Some(layer),
    //     _ => None,
    // }).enumerate() {
    //     layer.set_params(loaded_weights[i].clone(), loaded_bias[i].clone());
    // }

    // ---------------------TRAINING----------------------
    for iteration in 0..=150 {
        let (loss, accuracy) = nn.train_step(&x, &y, iteration, 150, "../test_data/mnist");
        
        if iteration%10 == 0 {
            println!("Iteration {}: loss = {}, accuracy = {}%", iteration, loss, accuracy);
        }
    }

    // ----------------TESTING & ACCURACY-----------------
    let (xt, yt) = load_testing_data()?;
    let num_layers = nn.layers
        .iter()
        .filter(|l| matches!(l, LayerTypes::Layer(_)))
        .count();
    let (loaded_weights, loaded_bias) = load_model_weights_and_biases("../test_data/mnist", num_layers)?;

    for (i, layer) in nn.layers.iter_mut().filter_map(|l| match l {
        LayerTypes::Layer(layer) => Some(layer),
        _ => None,
    }).enumerate() {
        layer.set_params(loaded_weights[i].clone(), loaded_bias[i].clone());
    }

    let (loss, accuracy) = nn.test_step(&xt, &yt);
    println!("Loss: {loss}");
    println!("Accuracy: {accuracy}%");

    Ok(())
}
