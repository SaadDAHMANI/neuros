// This code demonstrates how to utilize the Neuros crate to predict the output 
// of the function f(x,y) = sin(x*y).

use ndarray::{Ix2, Axis}; 
use linfa::Dataset;


use ndarray_csv::Array2Reader;
use neuros::activations::Activations;
use neuros::trainer::*;


#[allow(dead_code)]
pub fn ann_sin_test(){

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
           
                    
            //split the dataset into (80% learning, 20% testing) 
            let split_ratio : f32 = 0.8;            
            
            let (train_set, test_set) = dataset.split_with_ratio(split_ratio);
           
            // Give the ANN structure. 
            let mut ann : Evonet = Evonet::empty();
            ann.add_layer(Layer::new(records.dim().1, Activations::Sigmoid));
            ann.add_layer(Layer::new(4, Activations::Sigmoid));
            ann.add_layer(Layer::new(targets.dim().1, Activations::Linear));

            let train_algo : TrainingAlgo = TrainingAlgo::EO(EoSettings::default());

            let training_result = ann.do_learning(& train_algo, &train_set);

            println!("Training results : {:?}", training_result);

            let learning_out = ann.compute_outputs(&train_set); 

            match learning_out {
                Err(eror)=> println!("No result, due to : {}", eror),
                Ok(ann_out)=>{
                    
                    for (computed, expected) in ann_out.iter().zip(train_set.targets().iter()) {
                        println!("Computed: {:?}, Expected : {:?}", computed, expected);
                    } 
                }
            }


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
