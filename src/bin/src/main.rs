use neuros::{trainer::{Evonet, TrainerParams}, activations::Activations};

extern crate neuros;
use linfa::dataset::Dataset;
use ndarray::{Ix1, array};
use sefar::sequential_algos::eo::EOparams;


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

            // define arameters for the training (learning algorithm) 
            let population_size : usize =10;
            let dimensions: usize = ann.get_weights_biases_count();
            let max_iterations : usize = 10;
            let lb = vec![-5.0; dimensions];
            let ub = vec![5.0; dimensions];
            let a1 : f64 = 2.0;
            let a2 : f64 = 2.0;
            let gp :f64 = 0.5;

            let eoparams : EOparams = EOparams::new(population_size, dimensions, max_iterations, &lb, &ub, a1, a2, gp);  
            let trainer_params = TrainerParams::EoParams(eoparams); 
            ann.do_learning(&trainer_params);
        }
    }
    

}
