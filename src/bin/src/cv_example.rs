// This code demonstrates how to utilize the Neuros crate to predict the output 
// of the function f(x,y) = sin(x*y).

use ndarray::{Ix2, Axis}; 
use linfa::Dataset;


use ndarray_csv::Array2Reader;
use neuros::activations::Activations;
use neuros::trainer::*;


#[allow(dead_code)]
pub fn ann_sin_test_cross_valid(){

    let path = "src/bin/src/data/sin_xy.csv";
    //--------------------------
    let data = read_csv_file(path);

    match data{
        Err(eror) => println!("I can not read the given file because of :{:?}", eror),
        Ok(data)=> {
                        
            // Split the data to take the first two columns as inputs and the third (last) column as output.
            // records : inputs,
            // targets : outputs.
            let (records, targets) = data.view().split_at(Axis(1), 2);
            
            ////Create a data set from (inputs, outputs) samples
            let dataset : Dataset<f64, f64, Ix2> = Dataset::new(records.to_owned(),targets.to_owned());
                             
            // Give the ANN structure. 
            let mut ann : Evonet = Evonet::empty();
            ann.add_layer(Layer::new(records.dim().1, Activations::Sigmoid));
            ann.add_layer(Layer::new(3, Activations::Sigmoid));
            ann.add_layer(Layer::new(3, Activations::Sigmoid));
            ann.add_layer(Layer::new(targets.dim().1, Activations::Linear));

            let params : GoSettings = GoSettings::default();
            let train_algo : TrainingAlgo = TrainingAlgo::GO(params);
            
            let x = dataset.view().fold(4).into_iter().map(|(train, valid)| {
                println!(" --- Train : {:?}; valid : {:?}", train.targets.dim(), valid.targets.dim());

                let training_result = ann.clone().do_learning(&train_algo, &dataset);

                println!("Training results RMSE : {:?}", training_result.best_fitness);  
  
               let train_out = ann.compute_outputs(&train);
               let valid_out = ann.compute_outputs(&valid);
                         
            });

            println!("x : {:?}", x);
        },
    }


}

///
/// Read data from CSV file
/// path : csv file path 
/// 
fn read_csv_file(path : &str)-> Result<ndarray::Array2<f64>, Box<dyn std::error::Error>>{
    // Read an array back from the file
    let file = std::fs::File::open(path)?;
    let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(file);
    //let array_read: Array2<u64> = reader.deserialize_array2((2, 3))?;
    let array_read: ndarray::Array2<f64> = reader.deserialize_array2_dynamic()?;
    Ok(array_read)
}
