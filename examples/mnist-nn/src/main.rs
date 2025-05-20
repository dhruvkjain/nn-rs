use polars::prelude::*;
use std::error::Error;
use ndarray::Array2;
use model::NN;
use layers::layer::Layer;
use layers::relu::ReLu;
use loss::mseloss::MSELoss;
use optimizer::sgd::SGDOptimizer;

pub fn load_training_data() -> Result<(Array2<f32>, Array2<f32>), Box<dyn Error>> {
    let q = LazyCsvReader::new("../test_data/mnist/mnist_train.csv")
    .with_has_header(true)
    .finish()?;

    // let df = q.clone().with_streaming(true).collect()?;
    let training_labels = q
        .clone()
        .with_streaming(true)
        .select([col("label")])
        .collect()?;

    let training_data = q
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

    // taking transpose  
    traning_data_ndarray = traning_data_ndarray / 255.0;
    training_labels_ndarray = training_labels_ndarray;

    let data_dimensions:&[usize] = traning_data_ndarray.shape();
    let labels_dimensions:&[usize] = training_labels_ndarray.shape();

    // println!("{}", traning_data_ndarray);
    // println!("{}", training_labels_ndarray);
    println!("DATA: {}, {}", data_dimensions[0], data_dimensions[1]);
    println!("LABELS: {}, {}", labels_dimensions[0], labels_dimensions[1]);
    Ok((traning_data_ndarray, training_labels_ndarray))
}

fn main() -> Result<(), Box<dyn Error>>{
    let (x, y) = load_training_data()?;

    let mut trainer = NN {
        layers: vec![
            Box::new(Layer::new(784, 128)),
            Box::new(ReLu::new()),
            Box::new(Layer::new(128, 10)),
        ],
        loss_fn: MSELoss { targets: None },
        optim: SGDOptimizer { lr: 0.01 },
    };
    
    for iteration in 0..251 {
        let loss = trainer.train_step(&x, &y);
        
        if iteration%50 == 0{
            println!("Iteration {}: loss = {}", iteration, loss);
        }
    }
    Ok(())
}
