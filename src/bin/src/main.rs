extern crate neuros;
extern crate linfa;
extern crate ndarray;
extern crate sefar;

use ndarray::{array, Ix2};
use linfa::dataset::Dataset;

//--------------------------------------------------------------------------
use neuros::{trainer::{Evonet, TrainerParams}, activations::Activations};
use sefar::algos::eo::EOparams;
use sefar::algos::pso::PSOparams;
use sefar::algos::go::GOparams;
//--------------------------------------------------------------------------
fn main() {
    println!("Hello, NEUROS!");
    
    ann_test_xor();
}

#[allow(dead_code)]
fn ann_test_xor(){

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
 
     // shuffle the dataset
     let shuffle : bool = true;
 
     //split the dataset into (80% learning, 20% testing) 
     let split_ratio : f32 = 0.8;
 
     //Create an artificial neural network using the given parameters.
     let mut ann_restult = Evonet::new(&layers, &activations, &dataset, shuffle, split_ratio);
 
     match &mut ann_restult{
         Err(error) => panic!("Finish due to error : {}", error),
         Ok(ann)=>{
             
             // Train the neural network using Equilibrium Optimizer (EO):
             test_eo_trainer(ann);

            println!("_________________________________________________________________________");
 
             // Train the neural network using Particle Swarm Optimizer (PSO):
             test_pso_trainer(ann);
             println!("________________________________________________________________________");

             // Train the neural network using Growth Optimizer (EO):
             test_go_trainer(ann);
             println!("________________________________________________________________________");
         }
     }
}

///
/// Run training using Growth Optimizer (GO).
///   
#[allow(dead_code)]
fn test_go_trainer(ann : &mut Evonet){

    println!("___________ANN trained by Growth Optimizer (GO)__________________");

    // define parameters for the training (learning algorithm) 
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
   
   println!("Learning results : RMSE_Learning = {:?}", learning_results.best_fitness);

   let x = ann.do_testing();
   println!("Testing results = {:?}", x);

   // Compute the output for a given input:
   let sample = &[0.0, 0.0]; 

   match ann.compute(sample){
           Err(eror) => println!("There is an error due to : {}", eror),
           Ok(ann_out) =>{
               println!("The ANN output for {:?} is : {:?}", sample, ann_out);
           },
   };                       

}

///
/// Run training using Equilibrium Optimizer (EO).
///   
#[allow(dead_code)]
fn test_eo_trainer(ann : &mut Evonet){

    println!("___________ANN trained by Equilibrium Optimizer (EO)__________________");

    // define arameters for the training (learning algorithm) 
    let population_size : usize = 50; // set the search poplation size,
    let dimensions: usize = ann.get_weights_biases_count(); // get the search space dimension, 
    let max_iterations : usize = 500; // set the maximum number of iterations (learning step),
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

             // Compute the output for a given input:
            let sample = &[0.0, 0.0]; 

            match ann.compute(sample){
                    Err(eror) => println!("There is an error due to : {}", eror),
                    Ok(ann_out) =>{
                        println!("The ANN output for {:?} is : {:?}", sample, ann_out);
                    },
            };                     

        }
    }  
}

///
/// Run training using Particle Swarm Optimizer (PSO).
///   
#[allow(dead_code)] 
fn test_pso_trainer(ann : &mut Evonet){
    println!("___________ANN trained by Particle Swarm Optimizer (PSO)__________________");

    // define arameters for the training (learning algorithm) 
    let population_size : usize = 50; // set the search poplation size,
    let dimensions: usize = ann.get_weights_biases_count(); // get the search space dimension, 
    let max_iterations : usize = 500; // set the maximum number of iterations (learning step),
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
             // Compute the output for a given input:
            let sample = &[0.0, 0.0]; 

            match ann.compute(sample){
                    Err(eror) => println!("There is an error due to : {}", eror),
                    Ok(ann_out) =>{
                        println!("The ANN output for {:?} is : {:?}", sample, ann_out);
                    },
            }; 
        }
    }  
}