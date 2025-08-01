use std::error::Error;

use ndarray::{Array1, Array2};
use polars::prelude::*;

pub fn load_array2_from_csv(save_path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    let lf = LazyCsvReader::new(save_path)
        .with_has_header(true)
        .finish()?;
    
    let df = lf
        .with_streaming(true)
        .cast_all(DataType::Float32, true)
        .collect()?;

    let cols = df.get_columns();

    let height = cols[0].len();
    let width = cols.len();

    let mut data = Vec::with_capacity(height * width);
    for row in 0..height {
        for col in 0..width {
            data.push(cols[col].f32()?.get(row).unwrap());
        }
    }

    Ok(Array2::from_shape_vec((height, width), data)?)
}

pub fn load_array1_from_csv(save_path: &str) -> Result<Array1<f32>, Box<dyn Error>> {
    let lf = LazyCsvReader::new(save_path)
        .with_has_header(true)
        .finish()?;
    
    let df = lf
        .with_streaming(true)
        .cast_all(DataType::Float32, true)
        .collect()?;

    let col = &df.get_columns()[0];
    let data = col.f32()?.into_no_null_iter().collect::<Vec<_>>();
    Ok(Array1::from(data))
}

pub fn load_model_weights_and_biases(save_path: &str, num_hidden_layers: usize) -> Result<(Vec<Array2<f32>>, Vec<Array1<f32>>), Box<dyn Error>> {
    let mut all_weights = Vec::new();
    let mut all_bias = Vec::new();

    for idx in 0..num_hidden_layers {
        let w_path = format!("{}/w{}.csv", save_path, idx);
        let b_path = format!("{}/b{}.csv", save_path, idx);

        let weight = load_array2_from_csv(&w_path)?;
        let bias = load_array1_from_csv(&b_path)?;

        all_weights.push(weight);
        all_bias.push(bias);
    }

    Ok((all_weights, all_bias))
}