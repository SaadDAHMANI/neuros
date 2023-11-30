use std::option::Iter;

use super::activations::*;
use super::neuralnet::*;

extern crate linfa;
use linfa::dataset::Dataset;

extern crate ndarray;
use ndarray::Ix1;

use rand::prelude::*;

use sefar::sequential_algos::eo;
use sefar::sequential_algos::eo::*;
use sefar::core::genome::*;
use sefar::core::objectivefunction::Objectivefunction;
use sefar::core::optimization_result::OptimizationResult;

pub struct Evonet<'a> {
    // max_iter : usize;
    pub dataset : &'a Dataset<f64, f64, Ix1>,
    pub k_fold : Option<usize>,
    pub activations: &'a Vec<Activations>,
	pub layers: &'a Vec<usize>,
    neuralnetwork : Neuralnet,
    learning_set : Option<Dataset<f64, f64, Ix1>>,
    testing_set :  Option<Dataset<f64, f64, Ix1>>,
}

impl<'a> Evonet<'a> {

    #[allow(dead_code)]
    pub fn new(layers: &'a Vec<usize>, activations :&'a Vec<Activations>, dataset : &'a Dataset<f64, f64, Ix1>, k_fold : Option<usize>, shuffle: bool, split_ratio : f32)-> Result<Evonet<'a>, String>{
        
        if layers.len() < 2 {
          return Err("Layers must be greater than 1.".to_owned());
        }

        if activations.len() != layers.len() {
            return Err("Lenghts of Activations and Layers must be equals.".to_owned());
        }

        let tmp_dataset = dataset.clone();
         //shuffl and split data 
         if shuffle {
            let mut rng = rand::thread_rng();
            tmp_dataset.shuffle(&mut rng);
        }

        let (training_set, testing_set) = tmp_dataset.split_with_ratio(split_ratio);
                
        let nnet : Neuralnet = Neuralnet::new(layers.clone(), activations.clone());
        
        Ok (Evonet {
           // max_iter : iterations,
            dataset,
            k_fold,
            activations,
            layers,
            neuralnetwork: nnet,
            learning_set : Some(training_set),
            testing_set : Some(testing_set),
        })
    }

    pub fn get_weights_biases_count(&self)->usize{
        self.neuralnetwork.get_weights_biases_count()
    }

    ///
    /// perform supervised learning.
    ///     
    #[allow(dead_code)]
    pub fn do_learning(&mut self, params : &EOparams) -> OptimizationResult {
        let wb = self.neuralnetwork.get_weights_biases_count();
        let lb = vec![-5.0; wb]; //Vec::new();
        let ub = vec![5.0; wb];
        
        let newparams = EOparams {
            population_size : params.population_size,           
            max_iterations : params.max_iterations,
            dimensions  : wb,
            lower_bounds : &lb,
            upper_bounds : &ub,
            a1 : params.a1,
            a2 : params.a2,
            gp : params.gp,
        };

                     
        let result  = sefar::sequential_algos::eo::eo(&newparams, self);
        result
    }
    
    #[allow(dead_code)]
     fn convert22dvec(ds : &Vec<f64>)->Vec<Vec<f64>>{
        
        let mut result = vec![vec![0.0f64; 1]; ds.len()];
        
        let mut i : usize = 0;
        for itm in ds.iter() {
            result[i][0] = itm.clone();
            i+=1;    
        };
                
        result
    }
}    
  
impl<'a> Objectivefunction for Evonet<'a>{
    fn evaluate(&mut self, genome : &Vec<f64>)->f64 {
                         
        //let kf : usize = match self.k_fold {
        //        None =>  1,
        //        Some(kf) =>  kf,
        // };
         
        //let kfolds = self.dataset.fold(kf);  

         //1. Update weights and biases :
         self.neuralnetwork.update_weights_biases(genome);
          
         let learning_err = match &self.learning_set {
             None => f64::NAN,
             Some(train_set)=>{
                
                
                0.0f64
             }, 
         }; 
               
         learning_err 
    }
    
   

}




