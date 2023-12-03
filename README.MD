# Neuros

[Neuros](https://github.com/SaadDAHMANI/neuros) is a [Rust](https://www.rust-lang.org/) package for Artificial (Feedforward) Neural Networks processing.

## Training algorithms 
The current version includes:

- Equilibrium Optimizer (EO), 
- Particle Swarm Optimizer (PSO).

## How to use NEUROS
```rust
use neuros::{trainer::{Evonet, TrainerParams}, activations::Activations};
extern crate neuros;
use linfa::dataset::Dataset;
use ndarray::{Ix1, array};
use sefar::sequential_algos::{eo::EOparams, pso::PSOparams};

fn main() {
    println!("Hello, NEUROS!");

    // Give the ANN structure. 
    let layers = [2, 3, 1].to_vec();

    //Give activation function for each layer.
    let activations = [Activations::Sigmoid, Activations::Sigmoid, Activations::Sigmoid].to_vec();

    //Give input data samples
    let records = array![[1.,1.], [2.,1.], [3.,2.], [4.,1.],[5., 3.], [6.,2.]];
   
    //Give output data samples
    let targets = array![1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    
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
            test_eo_trainer(ann);

            // run pso_trainer
            test_pso_trainer(ann);
        }
    }   
}

///
/// Run training using Equilibrium Optimizer.
///   
pub fn test_eo_trainer(ann : &mut Evonet){
    // define arameters for the training (learning algorithm) 
    let population_size : usize = 100; // set the search poplation size,
    let dimensions: usize = ann.get_weights_biases_count(); // get the search space dimension, 
    let max_iterations : usize = 1000; // set the maximum number of iterations (learning step),
    let lb = vec![-5.0; dimensions]; // set the lower bound for the search space,
    let ub = vec![5.0; dimensions]; // set the upper bound for the search space,
    let a1 : f64 = 2.0; // give the value of a1 parameter (Equilibrium Optimizer),
    let a2 : f64 = 2.0; // give the value of a2 parameter (Equilibrium Optimizer),
    let gp :f64 = 0.5; // give the value of GP parameter (Equilibrium Optimizer),

    // create the EO parameters (learning algorithm)
    let params : EOparams = EOparams::new(population_size, dimensions, max_iterations, &lb, &ub, a1, a2, gp);  
    let trainer_params = TrainerParams::EoParams(params); 
    
    // perform the learning step. 
   let learning_results = ann.do_learning(&trainer_params);

   println!("RMSE_Learning = {:?}", learning_results.best_fitness);
}

///
/// Run training using Equilibrium Optimizer.
///   
pub fn test_pso_trainer(ann : &mut Evonet){
    // define arameters for the training (learning algorithm) 
    let population_size : usize = 100; // set the search poplation size,
    let dimensions: usize = ann.get_weights_biases_count(); // get the search space dimension, 
    let max_iterations : usize = 1000; // set the maximum number of iterations (learning step),
    let lb = vec![-5.0; dimensions]; // set the lower bound for the search space,
    let ub = vec![5.0; dimensions]; // set the upper bound for the search space,
    let c1 : f64 = 2.0; // give the value of a1 parameter (EPSO),
    let c2 : f64 = 2.0; // give the value of a2 parameter (PSO),
    
    // create the EO parameters (learning algorithm)
    let params : PSOparams = PSOparams::new(population_size, dimensions, max_iterations, &lb, &ub, c1, c2);  
    let trainer_params = TrainerParams::PsoParams(params); 
    
    // perform the learning step. 
   let learning_results = ann.do_learning(&trainer_params);

   println!("RMSE_Learning = {:?}", learning_results.best_fitness);
}
```

 
