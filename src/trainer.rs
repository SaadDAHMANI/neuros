use std::option::Iter;

use super::activations::*;
use super::neuralnet::*;

extern crate linfa;
use linfa::dataset::Dataset;

extern crate ndarray;
use ndarray::Ix1;

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
}

impl<'a> Evonet<'a> {
    pub fn new (layers: &'a Vec<usize>, activations :&'a Vec<Activations>, dataset : &'a Dataset<f64, f64, Ix1>, k_fold : Option<usize>)-> Result<Evonet<'a>, String>{
        
        if layers.len() < 2 {
          return Err("Layers must be greater than 1.".to_owned());
        }

        if activations.len() != layers.len() {
            return Err("lenghts of Activations and Layers must be equals.".to_owned());
        }

        let nnet : Neuralnet = Neuralnet::new(layers.clone(), activations.clone());
        Ok (Evonet {
           // max_iter : iterations,
            dataset,
            k_fold,
            activations,
            layers,
            neuralnetwork: nnet,
        })
    }

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
    
     fn convert22dvec(ds : &Vec<f64>)->Vec<Vec<f64>>{
        
        let result = vec![vec![0.0f64; 1]; ds.len()];
        
        let mut i : usize = 0;
        for itm in ds.iter() {
            result[i][0] = itm.clone();    
        };
                
        result
    }
}    
  
impl<'a> Objectivefunction for Evonet<'a>{
    fn evaluate(&mut self, genome : &Vec<f64>)->f64 {
        //         
        let kf : usize = match self.k_fold {
                None =>  1,
                Some(kf) =>  kf,
         };
         
        let kfolds = self.dataset.fold(kf);
        
        for onefold in kfolds {
            
            //1. Update weights and biases :
            self.neuralnetwork.update_weights_biases(genome);
            
            // for rd in onefold.0.records{
                
            //}
            
            
            //2. Compute RMSE 
            //let rmse = self.neuralnetwork.compute_learning_error_rmse(
            //                &onefold.0.records,
            //                &onefold.0.targets.to_vec());        
            
            
            
        }      
        
        
        0.0f64   
    }
    
   

}




