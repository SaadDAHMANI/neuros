
use super::activations::*;
use super::neuralnet::*;

extern crate linfa;
use linfa::Dataset;

use ndarray::{Ix1,Ix2};

extern crate sefar;
use sefar::core::eoa::EOA;
use sefar::core::{problem::Problem, optimization_result::OptimizationResult};
use sefar::algos::go::{GO, GOparams};

///
/// Parameters of the training algorithm.
/// EoParams(PSOparams<'a>): Parameters for Equilibrium Optimizer,
/// PsoParams(PSOparams<'a>): Parameters for Particle Swarm Optimizer,
pub enum TrainerParams<'a>{
    //EoParams(EOparams<'a>),
    //PsoParams(PSOparams<'a>),
    GoParams(GOparams<'a>),
}

#[derive(Debug, Clone)]
pub struct Evonet<'a> {
    // max_iter : usize;
    pub dataset : &'a Dataset<f64, f64, Ix2>,
    pub k_fold : Option<usize>,
    pub activations: &'a Vec<Activations>,
	pub layers: &'a Vec<usize>,
    neuralnetwork : Neuralnet,
    learning_set : Dataset<f64, f64, Ix2>,
    testing_set :  Dataset<f64, f64, Ix2>,

    /// number of records (i.e., of samples).
    record_count : usize,

    /// number of features in records.
    record_features : usize,

    /// number of targets (i.e., of samples).
    target_count : usize,

    /// number of features in targets.
    target_features : usize,
}

impl<'a> Evonet<'a> {

    #[allow(dead_code)]
    pub fn new(layers: &'a Vec<usize>, activations :&'a Vec<Activations>, dataset : &'a Dataset<f64, f64, Ix2>, k_fold : Option<usize>, shuffle: bool, split_ratio : f32)-> Result<Evonet<'a>, String>{
        
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

        let (record_count, record_features) = dataset.records.dim();
        let (target_count, target_features) = dataset.targets.dim();
       
        Ok (Evonet {
           // max_iter : iterations,
            dataset,
            k_fold,
            activations,
            layers,
            neuralnetwork: nnet,
            learning_set : training_set,
            testing_set : testing_set,
            record_count,
            record_features,
            target_count, 
            target_features,
        })
    }

    pub fn get_weights_biases_count(&self)->usize{
        self.neuralnetwork.get_weights_biases_count()
    }

    ///
    /// perform supervised learning.
    ///     
    #[allow(dead_code)]
    pub fn do_learning(&mut self, params : &TrainerParams) -> OptimizationResult {
        //let wb = self.neuralnetwork.get_weights_biases_count();
        //let lb = vec![-5.0; wb]; //Vec::new();
        //let ub = vec![5.0; wb];
        
      let result =  match params{
            // TrainerParams::EoParams(params) => {
            //     sefar::sequential_algos::eo::eo(&params, self)
            // },

            // TrainerParams::PsoParams(params)=>{
            //     sefar::sequential_algos::pso::pso(&params, self)
            // },

            TrainerParams::GoParams(params) => {
                  let mut algo = GO::<Evonet>::new(params, self);
                  let result = algo.run();
                  result
            },
        };       
                     
        //
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

impl<'a> Problem for Evonet<'a> {
    fn objectivefunction(&mut self, genome : &[f64])->f64 {
                       
         //1. Update weights and biases :
        self.neuralnetwork.update_weights_biases(genome);
        
        let mut errors : Vec<f64> = vec![0.0; self.target_features];

        for (x, y) in self.learning_set.sample_iter(){
            match x.as_slice() {
                None=>{},
                Some(x_vector)=> {
                    let computed =  self.neuralnetwork.feed_forward(x_vector);
                        match y.as_slice() {
                            None => {},
                            Some(y_vector)=> {                                  
                                for j in 0..self.target_features{
                                    errors[j] += (y_vector[j] - computed[j]).powi(2);
                                }                            
                            },
                        };
                    },
                }
            };
            
            // compute RMSE for learning samples for each output
            for i in 0..self.target_features {
                errors[i] = f64::sqrt(errors[i]/self.target_features as f64);
            }              
        
        //learning_err = sum of RMSE errors:
        let rmse_error : f64 = errors.iter().fold(0.0f64, |sum, a| sum + a);
        rmse_error
    }
    


   

}




