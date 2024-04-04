extern crate neuros;
extern crate ndarray;
extern crate linfa;
extern crate csv;
extern crate ndarray_csv;
extern crate sefar;

//-------------------------------------------------------------------------
use linfa::dataset::Dataset;
//-------------------------------------------------------------------------
use csv::ReaderBuilder;
use ndarray::{array, Array2, Axis, Ix2};
use ndarray_csv::Array2Reader;
use core::panic;
use std::error::Error;
use std::fs::File;
//--------------------------------------------------------------------------
use neuros::{trainer::{Evonet, TrainerParams}, activations::Activations};
use sefar::algos::go::GOparams;
use sefar::algos::eo::EOparams;
use sefar::algos::pso::PSOparams;
//--------------------------------------------------------------------------

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
     let targets = array![[0.], [1.], [1.], [0.]];
     
     //Create a data set from (inputs, outputs) samples
     let dataset : Dataset<f64, f64, Ix2> = Dataset::new(records, targets);
 
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
             test_eo_trainer(ann);
 
             // run pso_trainer
              test_pso_trainer(ann);
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
            
            ////Create a data set from (inputs, outputs) samples
            let dataset : Dataset<f64, f64, Ix2> = Dataset::new(records.to_owned(),targets.to_owned());
           
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
                    
                    // Run Growth optimizer trainer for the given ANN.
                    test_go_trainer(ann);                
                    
                }
            };
        },
    };

}


///
/// Run training using Growth Optimizer (GO).
///   
#[allow(dead_code)]
fn test_go_trainer(ann : &mut Evonet){
    // define arameters for the training (learning algorithm) 
    let population_size : usize = 50; // set the search poplation size,
    let dimensions: usize = ann.get_weights_biases_count(); // get the search space dimension, 
    let max_iterations : usize = 500; // set the maximum number of iterations (learning step),
    let lb = vec![-5.0; dimensions]; // set the lower bound for the search space,
    let ub = vec![5.0; dimensions]; // set the upper bound for the search space,

    // create the EO parameters (learning algorithm)
    let params : GOparams = GOparams {
        population_size,
        dimensions,
        max_iterations,
        lower_bounds: &lb, 
        upper_bounds : &ub,
    };  

    let trainer_params = TrainerParams::GoParams(params); 
    
    // perform the learning step. 
   let learning_results = ann.do_learning(&trainer_params);

   println!("RMSE_Learning = {:?}", learning_results.best_fitness);
}

///
/// Run training using Equilibrium Optimizer (EO).
///   
#[allow(dead_code)]
fn test_eo_trainer(ann : &mut Evonet){
    // define arameters for the training (learning algorithm) 
    let population_size : usize = 100; // set the search poplation size,
    let dimensions: usize = ann.get_weights_biases_count(); // get the search space dimension, 
    let max_iterations : usize = 1000; // set the maximum number of iterations (learning step),
    let lb = vec![-5.0; dimensions]; // set the lower bound for the search space,
    let ub = vec![5.0; dimensions]; // set the upper bound for the search space,
    let a1 : f64 = 2.0; // give the value of a1 parameter (Equilibrium Optimizer),
    let a2 : f64 = 1.0; // give the value of a2 parameter (Equilibrium Optimizer),
    let gp :f64 = 0.5; // give the value of GP parameter (Equilibrium Optimizer),

    // create the EO parameters (learning algorithm)
    let params = EOparams::new(population_size, dimensions, max_iterations, &lb, &ub, a1, a2, gp);  
    
    match params {
        Err(eror) => println!("I can not run because this error : {:?}", eror),
        Ok(settings)=> {
            let trainer_params = TrainerParams::EoParams(settings); 
             // perform the learning step. 
            let learning_results = ann.do_learning(&trainer_params);
            println!("EO trainer : RMSE_Learning = {:?}", learning_results.best_fitness);
        }
    }  
}

///
/// Run training using Particle Swarm Optimizer (PSO).
///   
#[allow(dead_code)] 
fn test_pso_trainer(ann : &mut Evonet){
    // define arameters for the training (learning algorithm) 
    let population_size : usize = 100; // set the search poplation size,
    let dimensions: usize = ann.get_weights_biases_count(); // get the search space dimension, 
    let max_iterations : usize = 1000; // set the maximum number of iterations (learning step),
    let lb = vec![-5.0; dimensions]; // set the lower bound for the search space,
    let ub = vec![5.0; dimensions]; // set the upper bound for the search space,
    let c1 : f64 = 2.0; // give the value of a1 parameter (PSO),
    let c2 : f64 = 2.0; // give the value of a2 parameter (PSO),
    
    // create the PSO parameters (learning algorithm)
    let params  = PSOparams::new(population_size, dimensions, max_iterations, &lb, &ub, c1, c2);  
    
    match params {
        Err(eror) => println!("I can not run because this error : {:?}", eror),
        Ok(settings)=> {
            let trainer_params = TrainerParams::PsoParams(settings); 
             // perform the learning step. 
            let learning_results = ann.do_learning(&trainer_params);
            println!("PSO trainer : RMSE_Learning = {:?}", learning_results.best_fitness);
        }
    }  
}



