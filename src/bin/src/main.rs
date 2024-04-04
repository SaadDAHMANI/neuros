extern crate neuros;
extern crate ndarray;
extern crate linfa;
extern crate csv;
extern crate ndarray_csv;

//-------------------------------------------------------------------------
use neuros::{trainer::{Evonet, TrainerParams}, activations::Activations};
use linfa::dataset::Dataset;
use ndarray::{Ix1, array};
//-------------------------------------------------------------------------
use sefar::sequential_algos::{eo::EOparams, pso::PSOparams};
//-------------------------------------------------------------------------
use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, OwnedRepr, Dim};
use ndarray_csv::{Array2Reader, Array2Writer};
use core::panic;
use std::error::Error;
use std::fs::File;
//----------------------------------------------------------------------------


fn main() {
    println!("Hello, NEUROS!");
    
    //Path of the data set:
    //let path = "src/bin/src/data/test_data.csv";

     //Path of the data set:
     let path = "src/bin/src/data/sin_xy.csv";

    ann_test_2(path);
         

}

#[allow(dead_code)]
fn ann_test_1(){

     // Give the ANN structure. 
     let layers = [2, 3, 1].to_vec();

     //Give activation function for each layer.
     let activations = [Activations::Sigmoid, Activations::Sigmoid, Activations::Sigmoid].to_vec();
 
     //Give input data samples
     let records = array![[0.,0.], [1.,0.], [0.,1.], [1.,1.]];
    
     //Give output data samples
     let targets = array![0., 1., 1., 0.];
     
     //Create a data set from (inputs, outputs) samples
     let dataset : Dataset<f64, f64, Ix1> = Dataset::new(records, targets);
 
     let k_fold : Option<usize> = Some(2);
 
     // shuffle the dataset
     let shuffle : bool = true;
 
     //split the dataset into (70% learning, 30% testing) 
     let split_ratio : f32 = 0.7;
 
     //Create an artificial neural network using the given parameters.
     let mut ann_restult = Evonet::new(&layers, &activations, &dataset, k_fold, shuffle, split_ratio);
 
     match &mut ann_restult{
         Err(error) => panic!("Finish due to error : {}", error),
         Ok(ann)=>{
             
             // run eo_trainer
             //test_eo_trainer(ann);
 
             // run pso_trainer
             //test_pso_trainer(ann);
         }
     }
}


///
/// Read data from CSV file
/// path : csv file path 
/// 
fn read_csv_file(path : &str)-> Result<Array2<f64>, Box<dyn Error>>{
    // Read an array back from the file
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    //let array_read: Array2<u64> = reader.deserialize_array2((2, 3))?;
    let array_read: Array2<f64> = reader.deserialize_array2_dynamic()?;

    Ok(array_read)
}

#[allow(dead_code)]
fn ann_test_2(path : &str){
    //--------------------------
    let data = read_csv_file(path);

    match data{
        Err(eror) => println!("I can not read the given file because of :{:?}", eror),
        Ok(data)=> {
                        
            // Split the data to take the first two columns as inputs and the third (last) column as output.
            // records : inputs,
            // targets : outputs.
            let (records, targets) = data.view().split_at(Axis(1), 2);

            let array2d: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = targets.to_owned();

             // Convert to 1D array
            let n : usize = array2d.len();

            let target: ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = match array2d.into_shape((n,)){
                Err(eror) => panic!("I can not convert target into 1D, because of : {:?}", eror),
                Ok(data)=> data,
            };

            ////Create a data set from (inputs, outputs) samples
            let dataset : Dataset<f64, f64, Ix1> = Dataset::new(records.to_owned(),target);

            let k_fold : Option<usize> = Some(2);
 
            // shuffle the dataset
            let shuffle : bool = true;
        
            //split the dataset into (70% learning, 30% testing) 
            let split_ratio : f32 = 0.8;
           
            // Give the ANN structure. 
            let layers = [2, 3, 1].to_vec();

            // //Give activation function for each layer.
            let activations = [Activations::Sigmoid, Activations::Sigmoid, Activations::Linear].to_vec();
 
            //Create an artificial neural network using the given parameters.
            let mut ann_restult = Evonet::new(&layers, &activations, &dataset, k_fold, shuffle, split_ratio);
            
            match &mut ann_restult{
                Err(error) => panic!("Finish due to error : {}", error),
                Ok(ann)=>{
                                       
                    // run eo_trainer
                    //test_eo_trainer(ann);
        
                    // run pso_trainer
                    //test_pso_trainer(ann);
                }
            };

        },
    };

}



