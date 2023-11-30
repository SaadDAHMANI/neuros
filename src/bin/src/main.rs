use neuros::{trainer::Evonet, activations::Activations};

extern crate neuros;
use linfa::dataset::Dataset;
use ndarray::{Ix1, array};
use sefar::sequential_algos::eo::EOparams;


fn main() {
    println!("Hello, world!");
    let layers = [2, 3, 1].to_vec();
    let activations = [Activations::Sigmoid, Activations::Sigmoid, Activations::Sigmoid].to_vec();

    let records = array![[1.,1.], [2.,1.], [3.,2.], [4.,1.],[5., 3.], [6.,2.]];
    let targets = array![1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    
    let dataset : Dataset<f64, f64, Ix1> = Dataset::new(records, targets);

    let k_fold : Option<usize> = Some(2);

    // shuffle the dataset
    let shuffle : bool = true;

    //split the dataset into (70% learning, 30% testing) 
    let split_ratio : f32 = 0.7;

    let mut ann_restult = Evonet::new(&layers, &activations, &dataset, k_fold, shuffle, split_ratio);

    match &mut ann_restult{
        Err(error) => panic!("Finish due to error : {}", error),
        Ok(ann)=>{
            let population_size : usize =10;
            let dimensions: usize = ann.get_weights_biases_count();
            let max_iterations : usize = 10;
            let lb = vec![-5.0; dimensions];
            let ub = vec![5.0; dimensions];
            let a1 : f64 = 2.0;
            let a2 : f64 = 2.0;
            let gp :f64 = 0.5;

            let params : EOparams = EOparams::new(population_size, dimensions, max_iterations, &lb, &ub, a1, a2, gp);  
            
            ann.do_learning(&params);
        }
    }
    

}
