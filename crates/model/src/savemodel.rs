use std::error::Error;
use std::fs::File;

use ndarray::{Array1, Array2};
use polars::prelude::*;

pub fn array2_to_data_frame(array: &Array2<f32>, name: &str, idx: usize) -> DataFrame {
    let cols = array.shape()[1];

    let mut columns: Vec<Column> = Vec::new();

    for i in 0..cols {
        let col: Vec<f32> = array.column(i).to_vec();
        let col_name = format!("{}{}_{}", name, idx, i);
        columns.push(Column::new(PlSmallStr::from(col_name), col));
    }

    DataFrame::new(columns).expect("Failed to create weights DataFrame")
}

pub fn array1_to_data_frame(array: &Array1<f32>, name: &str, idx: usize) -> DataFrame {
    let col_name = format!("{}{}", name, idx);
    let mut bias: Vec<Column> = Vec::new();
    bias.push(Column::new(PlSmallStr::from(col_name), array.to_vec()));
    DataFrame::new(bias).expect("Failed to create bias DataFrame")
}

pub fn save_model (all_weights: Vec<&mut Array2<f32>>, all_bias: Vec<&mut Array1<f32>>, save_path: &str) -> Result<(), Box<dyn Error>> {
    
    for (idx, weight) in all_weights.into_iter().enumerate() {
        let full_save_path = format!("{}/w{}.csv", save_path, idx);
        let mut file = File::create(full_save_path).expect("could not weight create file");
        
        let mut wdf = array2_to_data_frame(&weight, "w", idx);

        CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut wdf)
        .expect("Failed to write weight in csv");
    }

    for (idx, bias) in all_bias.into_iter().enumerate() {
        let full_save_path = format!("{}/b{}.csv", save_path, idx);
        let mut file = File::create(full_save_path).expect("could not bias create file");
        
        let mut bdf = array1_to_data_frame(&bias, "b", idx);

        CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut bdf)
        .expect("Failed to write bias in csv");
    }

    Ok(())
}