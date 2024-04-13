# Neuros

[Neuros](https://github.com/SaadDAHMANI/neuros) is a [Rust](https://www.rust-lang.org/) package for Artificial (Feedforward) Neural Networks (ANNs) processing. [Neuros](https://github.com/SaadDAHMANI/neuros) uses [Sefar](https://crates.io/crates/sefar) crate to perform ANNs training. 

In the learning (training) stage, [Neuros](https://github.com/SaadDAHMANI/neuros) minimizes the Root Mean Square Error (RMSE) between computed and given model outputs. 

$Error_{Learning} = RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ 

Where $y_i$ and $\hat{y}_i$ represent the computed and given model outputs, respectively.

In the case of multiple ANN outputs, [Neuros](https://github.com/SaadDAHMANI/neuros) minimizes the sum of RMSE.  

$Error_{Learning} = \sum RMSE$ 

## Training algorithms 
The current version suppoerts the following ANN training algorithms:

[x] Equilibrium Optimizer (EO), 

[x] Particle Swarm Optimizer (PSO),

[x] Growth optimizer (GO).

## How to use Neuros
Please, **check the folder [src/bin/src](https://github.com/SaadDAHMANI/neuros/tree/master/src/bin/src) for the examples**.

### Example (ANN with single output)

1. Import dependencies:

```toml
[dependencies]
neuros = "0.1.1"
ndarray ="0.15.6"
linfa ="0.7.0"
sefar ="0.1.3"
```
2. In the main.rs file: 

```rust
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

    // The code print something like :
    //Hello, NEUROS!
    //EO trainer : RMSE_Learning = Some(0.0001697399220037401)
    //PSO trainer : RMSE_Learning = Some(0.020424429066646356)
    //Growth Optimizer trainer : RMSE_Learning = Some(0.012549512112249629)
    //Growth Optimizer trainer: Testing results = [[0.9932959284463537]] 
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
 
     //split the dataset into (70% learning, 30% testing) 
     let split_ratio : f32 = 0.7;
 
     //Create an artificial neural network using the given parameters.
     let mut ann_restult = Evonet::new(&layers, &activations, &dataset, shuffle, split_ratio);
 
     match &mut ann_restult{
         Err(error) => panic!("Finish due to error : {}", error),
         Ok(ann)=>{
             
             // Train the neural network using Equilibrium Optimizer (EO):
             test_eo_trainer(ann);
 
             // Train the neural network using Particle Swarm Optimizer (PSO):
             test_pso_trainer(ann);

             // Train the neural network using Growth Optimizer (EO):
             test_go_trainer(ann);
         }
     }
}

///
/// Run training using Growth Optimizer (GO).
///   
#[allow(dead_code)]
fn test_go_trainer(ann : &mut Evonet){
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

   println!("Growth Optimizer trainer : RMSE_Learning = {:?}", learning_results.best_fitness);

   // perform testing using the test samples:
   let x = ann.do_testing();

   println!("Growth Optimizer trainer: Testing results = {:?}", x);
}

///
/// Run training using Equilibrium Optimizer (EO).
///   
#[allow(dead_code)]
fn test_eo_trainer(ann : &mut Evonet){
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
        }
    }  
}

///
/// Run training using Particle Swarm Optimizer (PSO).
///   
#[allow(dead_code)] 
fn test_pso_trainer(ann : &mut Evonet){
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
        }
    }  
}
```