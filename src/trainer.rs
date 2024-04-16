
use super::activations::*;
use super::neuralnet::*;

extern crate linfa;
use linfa::dataset::Records;
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
/// # Arguments 
/// 
#[derive(Debug, Clone)]
pub enum TrainerParams<'a>{

    /// Parameters for Equilibrium Optimizer.
    EoParams(EOparams<'a>),

    /// Parameters for Particle Swarm Optimizer.
    PsoParams(PSOparams<'a>),
    
    /// Parameters for Growth Optimizer.
    GoParams(GOparams<'a>),
}
/// A structure defining an Evolutionary Artificial Neural Network.
/// 
/// # Arguments
/// 
/// * `dataset`: The dataset containing learning and testing samples.
/// 
/// * `activations`: A list of activation functions corresponding to each layer.
/// 
/// * `layers`: The structure of the neural network. For example, `let layers = vec![3, 10, 1]` creates an Artificial Neural Network with 3 inputs, a hidden layer with 10 neurons, and 1 output.
 
#[derive(Debug, Clone)]
pub struct Evonet<'a> {
    /// The dataset including learning and testing samples. 
    pub dataset : &'a Dataset<f64, f64, Ix2>,

    //pub k_fold : Option<usize>,
    
    /// A list of Activation functions, according to each layer.
    pub activations: &'a Vec<Activations>,

    /// The structure of the neural network, 
    /// ex., [3, 10, 1] creates an Artificial Neural Network with 03 inputs,
    /// a hidden layer with 10 neurones and 01 output.
    ///  
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

    /// Create a new instance of Evonet.
    /// 
    /// # Arguments
    /// 
    /// * `layers`: A vector that specifies the structure of the artificial neural network (ANN).
    ///   For example, `layers = vec![4, 3, 2]` creates an ANN with 4 input neurons, 3 hidden neurons, and 2 output neurons.
    /// 
    /// * `dataset`: A dataset containing training and testing samples. The dataset will be split according to the `split_ratio` parameter.
    /// 
    /// * `shuffle`: A boolean indicating whether to shuffle the dataset.
    /// 
    /// * `split_ratio`: The ratio used to split the dataset into learning and testing samples. For instance, `split_ratio = 0.8` means using 80% of the dataset for learning and 20% for testing.
    ///     
    #[allow(dead_code)]
    pub fn new(layers: &'a Vec<usize>, activations :&'a Vec<Activations>, dataset : &'a Dataset<f64, f64, Ix2>, shuffle: bool, split_ratio : f32)-> Result<Evonet<'a>, String>{
        
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
            //k_fold,
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

    /// 
    /// Return the number of ANN weights and biases.
    /// 
    pub fn get_weights_biases_count(&self)->usize{
        self.neuralnetwork.get_weights_biases_count()
    }

    ///
    /// Conduct supervised learning utilizing the training portion of the dataset.
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
    
    ///
    /// Perform testing using the testing part of the given dataset  
    /// 
    pub fn do_testing(&mut self)->Vec<Vec<f64>>{
        let mut testing_result : Vec<Vec<f64>> = Vec::with_capacity(self.testing_set.nsamples());

        for (_x, _y) in self.testing_set.sample_iter(){
            match _x.as_slice(){
                None =>{},
                Some(x_vec) => {
                    let computed =  self.neuralnetwork.feed_forward(x_vec);
                    testing_result.push(computed);
                },
            };
        }
        testing_result
    } 

    ///
    /// Compute outputs for a given data inputs.
    /// 
    pub fn compute(&mut self, inputs : &[f64])-> Result<Vec<f64>, String> {
        if inputs.len() == self._record_features {
            Ok(self.neuralnetwork.feed_forward(inputs))
        }
        else {
            Err(String::from("The lenght of inputs must be equal to the length the neural network input"))
        }
    }

    

  /*   #[allow(dead_code)]
     fn convert22dvec(ds : &Vec<f64>)->Vec<Vec<f64>>{        
        let mut result = vec![vec![0.0f64; 1]; ds.len()];        
        let mut i : usize = 0;
        for itm in ds.iter() {
            result[i][0] = itm.clone();
            i+=1;    
        };                
        result
    } */

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




