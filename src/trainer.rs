
use super::activations::*;
use super::neuralnet::*;

extern crate linfa;
use linfa::Dataset;

use ndarray::Ix2;

extern crate sefar;
use sefar::core::eoa::EOA;
use sefar::core::{problem::Problem, optimization_result::OptimizationResult};
use sefar::algos::go::{GO, GOparams};
use sefar::algos::eo::{EO, EOparams};
use sefar::algos::pso::{PSO, PSOparams};

///
/// Parameters of the training algorithm.
/// EoParams(PSOparams<'a>): Parameters for Equilibrium Optimizer,
/// PsoParams(PSOparams<'a>): Parameters for Particle Swarm Optimizer,
pub enum TrainerParams<'a>{
    EoParams(EOparams<'a>),
    PsoParams(PSOparams<'a>),
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
    _record_count : usize,

    /// number of features in records.
    _record_features : usize,

    /// number of targets (i.e., of samples).
    _target_count : usize,

    /// number of features in targets.
    _target_features : usize,
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

        let (_record_count, _record_features) = dataset.records.dim();
        let (_target_count, _target_features) = dataset.targets.dim();
       
        Ok (Evonet {
           // max_iter : iterations,
            dataset,
            k_fold,
            activations,
            layers,
            neuralnetwork: nnet,
            learning_set : training_set,
            testing_set : testing_set,
            _record_count,
            _record_features,
            _target_count, 
            _target_features,
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
        let wb = self.neuralnetwork.get_weights_biases_count();
        //let lb = vec![-5.0; wb]; //Vec::new();
        //let ub = vec![5.0; wb];
                
      let result =  match params{
            // Use EO as trainer
            TrainerParams::EoParams(params) => {
                let mut settings = params.clone();
                settings.dimensions = wb;
                let mut algo = EO::<Evonet>::new(&settings, self);
                 algo.run()               
            },

            // Use POS as trainer
            TrainerParams::PsoParams(params)=>{
                let mut settings = params.clone();
                settings.dimensions = wb;
                let mut algo = PSO::<Evonet>::new(&settings, self);
                algo.run()
            },

            // Use GO as trainer
            TrainerParams::GoParams(params) => {
                let mut settings = params.clone();
                settings.dimensions = wb;
                let mut algo = GO::<Evonet>::new(&settings, self);
                algo.run()
                 
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
        
        let mut errors : Vec<f64> = vec![0.0; self._target_features];

        for (x, y) in self.learning_set.sample_iter(){
            match x.as_slice() {
                None=>{},
                Some(x_vector)=> {
                    let computed =  self.neuralnetwork.feed_forward(x_vector);
                        match y.as_slice() {
                            None => {},
                            Some(y_vector)=> {                                  
                                for j in 0..self._target_features{
                                    errors[j] += (y_vector[j] - computed[j]).powi(2);
                                }                            
                            },
                        };
                    },
                }
            };
            
            // compute RMSE for learning samples for each output
            for i in 0..self._target_features {
                errors[i] = f64::sqrt(errors[i]/self._target_features as f64);
            }              
        
        //learning_err = sum of RMSE errors:
        let rmse_error : f64 = errors.iter().fold(0.0f64, |sum, a| sum + a);
        rmse_error
    }
    


   

}




