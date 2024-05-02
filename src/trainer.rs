
use super::activations::*;
use super::neuralnet::*;

extern crate linfa;

//use linfa::dataset::Records;
use linfa::Dataset;
use ndarray::Ix2;

use std::default::Default; 
use std::fmt::Display;


extern crate sefar;
use sefar::core::eoa::EOA;
use sefar::core::{problem::Problem, optimization_result::OptimizationResult};
use sefar::algos::go::{GO, GOparams};
use sefar::algos::eo::{EO, EOparams};
use sefar::algos::pso::{PSO, PSOparams};


#[derive(Debug, Clone)]
pub struct Layer{
    /// The number of neurons in the layer. (not null).
   neurons : usize,
    
    /// The activation function of the layer.
   activation : Activations,
}

impl Layer {
    /// Return a new instance of the struct 'Layer'. 
    /// 'neurons' : The number of neurons in the layer. (not null).
    /// 'activation' : The activation function of the layer.
    pub fn new(neurons : usize, activation : Activations)-> Self{        
        Self{
            neurons : usize::max(neurons, 1),
            activation, 
        }
    }
    
    /// Return the number of neurons in the layer.
    pub fn get_neurons(&self)-> usize {
        self.neurons
    }

    /// Return the activation function of the layer. 
    pub fn get_activations(&self)-> Activations {
        self.activation
    }

    /// Update neuron count of the layer.
    pub fn update_neurons(&mut self, new_neuron_count : usize){
        self.neurons = usize::max(new_neuron_count,1);
    }

    /// Update the activation function of the layer.
    pub fn update_activation(&mut self, new_activation : Activations){
        self.activation = new_activation;
    }
}

///
/// Parameters of the training algorithm.
/// # Arguments 
/// 
#[derive(Debug, Clone)]
pub enum TrainingAlgo{
    EO(EoSettings),  
    PSO(PsoSettings),
    GO(GoSettings),
}


#[derive(Debug, Clone)]
pub struct Evonet<'a> {
    pub layers: Vec<Layer>,
    neuralnetwork : Neuralnet,
    learning_set : Option<&'a Dataset<f64, f64, Ix2>>,    
}

impl<'a> Evonet<'a> {
    /// Create a new instance of Evonet.
    /// 
    /// # Arguments
    /// 
    /// * `layers`: A vector that specifies the structure of the artificial neural network (ANN).
    ///     
    #[allow(dead_code)]
    pub fn new(layers : Vec<Layer>)-> Self {

        let mut ann_layers : Vec<usize> = Vec::with_capacity(layers.len());
        let mut activations : Vec<Activations> = Vec::with_capacity(layers.len());

        for l in layers.iter(){
            ann_layers.push(l.neurons); 
            activations.push(l.activation);
        }

        let neuralnetwork : Neuralnet = Neuralnet::new(ann_layers, activations);
        Self {
            layers,
            neuralnetwork,
            learning_set : None,         
        }
    }

    fn update_neuralnet(&mut self){
        let mut ann_layers : Vec<usize> = Vec::with_capacity(self.layers.len());
        let mut activations : Vec<Activations> = Vec::with_capacity(self.layers.len());

        for l in self.layers.iter(){
            ann_layers.push(l.neurons); 
            activations.push(l.activation);
        }

        self.neuralnetwork = Neuralnet::new(ann_layers, activations);
    }
    
    ///
    /// Create an empty neural network.
    /// 
    pub fn empty()-> Self{
        let layers : Vec<Layer> = Vec::new();
        let ann : Evonet = Evonet::new(layers);
        ann
    }

    ///
    /// Empty the layers of the neural network.
    /// 
    pub fn clear_layers(&mut self){
        self.layers.clear();
        self.update_neuralnet();
    }

    ///
    /// Add a single layer to the neural network.
    ///   
    pub fn add_layer(&mut self, layer : Layer){       
        self.layers.push(layer);
        self.update_neuralnet();       
    }
       
    ///
    /// Perform ANN training.
    /// 
    ///  # Arguments
    ///  
    /// * 'train_algo' : A training algorithm and its parameter.
    /// 
    /// * 'train_dataset' : A data set to train the ANN.
    ///  
    pub fn do_learning(&mut self, train_algo : &TrainingAlgo, train_dataset : &'a Dataset<f64, f64, Ix2>)-> OptimizationResult {
        self.learning_set = Some(train_dataset);
        let wb = self.neuralnetwork.get_weights_biases_count();
                
        let result =  match train_algo {
            // Use EO as trainer
            &TrainingAlgo::EO(stng) => {
                let lb = vec![stng.lower_bound; wb];
                let ub = vec![stng.upper_bound; wb];
                let settings = EOparams::new(stng.pop_size, wb, stng.max_iter, &lb, &ub, stng.a1, stng.a2, stng.gp);
                
                match settings {
                    Err(eror) => OptimizationResult::get_none(eror),
                    Ok(settings) => {
                        let mut algo = EO::<Evonet>::new(&settings, self);
                        algo.run()     
                    },
                }          
            },

            // Use POS as trainer
            &TrainingAlgo::PSO(stng)=>{
                let lb = vec![stng.lower_bound; wb];
                let ub = vec![stng.upper_bound; wb];
                let settings = PSOparams::new(stng.pop_size, wb ,stng.max_iter, &lb, &ub, stng.c1, stng.c2);
                match settings {
                    Err(eror) => OptimizationResult::get_none(eror),
                    Ok(settings) => {
                        let mut algo = PSO::<Evonet>::new(&settings, self);
                        algo.run()
                    },
                }               
            },

            // Use GO as trainer
            &TrainingAlgo::GO(stng) => {
                let lb = vec![stng.lower_bound; wb];
                let ub = vec![stng.upper_bound; wb];
                let settings : GOparams = GOparams::new(stng.pop_size, wb ,stng.max_iter, &lb, &ub);
           
                let mut algo = GO::<Evonet>::new(&settings, self);
                algo.run()
                 
            },
        };       
                     
        //
        result

    }

    pub fn compute_outputs(&mut self, dataset : &Dataset<f64, f64, Ix2>)-> Result<Vec<Vec<f64>>, crate::Error>{
        if dataset.records.dim().1 != self.neuralnetwork.layers[0]{
            Err(crate::Error::InvalidInputCount{expected: self.neuralnetwork.layers[0], actual: dataset.records.dim().1})
        } 
        else {
            let mut ann_out : Vec<Vec<f64>> = Vec::new();
            for (x, _) in dataset.sample_iter(){
                match x.as_slice(){
                    None =>{}, //Err(String::from("Not implemented yet!"))},
                    Some(x) => {
                        let y = self.compute_output(x);
                        match y {
                            Err(_err)=> {},
                            Ok(y) => ann_out.push(y),
                        }; 
                    },
                };
            };
            Ok(ann_out)
        }
    }

    pub fn compute_output(&mut self, inputs : &[f64])-> Result<Vec<f64>, String> {
        if self.layers[0].neurons == inputs.len(){
            Ok(self.neuralnetwork.feed_forward(inputs))
        }    
        else {    
            Err(String::from("The lenght of inputs is not equal the size of the neural network input layer!"))
        }         
    }

    /// 
    /// Return the number of ANN weights and biases.
    /// 
    pub fn get_weights_biases_count(&self)-> usize{
        self.neuralnetwork.get_weights_biases_count()
    }

    #[allow(dead_code)]
    pub fn save(&self, _file : &str)-> Result<(), String>{        
        Err(String::from("not implemented yet !!"))
    }

    #[allow(dead_code)]
    pub fn load(_file : &str)-> Result<Self, String>{
        Err(String::from("not implemented yet !!"))
    }

}

impl<'a> Problem for Evonet<'a> {
    fn objectivefunction(&mut self, genome : &[f64])-> f64 {

        match self.learning_set {
            None => f64::MAX,
            Some(learning_set)=> {
                let sample_count = learning_set.targets.dim().0 as f64 ;
                let target_features = learning_set.targets.dim().1;

                //println!("(i, j) = ({}, {})", sample_count, target_features);

                 //1. Update weights and biases :
                self.neuralnetwork.update_weights_biases(genome);

                let mut errors : Vec<f64> = vec![0.0; target_features];

                for (x, y) in learning_set.sample_iter(){
                    match x.as_slice() {
                        None=>{},
                        Some(x_vector)=> {
                            let computed =  self.neuralnetwork.feed_forward(x_vector);
                                match y.as_slice() {
                                
                                    None => {},
                                    Some(y_vector)=> {             
                                        //println!("(x, y): ({:?}, {:?})", x_vector, y_vector);

                                        for j in 0..target_features{
                                            errors[j] += f64::powi(y_vector[j] - computed[j], 2);
                                        }                            
                                   },
                            };
                        },
                    }
                };

                // compute RMSE for learning samples for each output
                for j in 0..target_features {
                    errors[j] = f64::sqrt(errors[j]/ sample_count);
                }              
        
                //learning_err = sum of RMSE errors:
                let rmse_error : f64 = errors.iter().fold(0.0f64, |sum, a| sum + a);
                rmse_error
            },
        }   
    }
}


/// Parameters for Equilibrium Optimizer (EO).
    /// 
    /// ## Example: 
    /// 
    /// let eo_params : EoSettings = EoSettings::new(50,500, -5.0, 5.0, 2.0, 1.0, 0.5); 
    /// 
    /// ### here,
    ///  
    /// * The population size (i.e., the count of search agents) = 50,
    /// 
    /// * The max iterations to perform by the training algorithm = 500,
    /// 
    /// * The lower bound for ANN weights and biases = -5.0, 
    /// 
    /// * The upper bound for ANN weights and biases = 5.0,
    /// 
    /// * a_1 = 2.0,
    /// 
    /// * a_2 = 1.0,
    /// 
    /// * gp = 0.5.
    /// 
#[derive(Debug, Clone, Copy)]
pub struct EoSettings{
    pub pop_size : usize,
    pub max_iter : usize,
    pub lower_bound : f64,
    pub upper_bound : f64,
    pub a1 : f64,
    pub a2 : f64,
    pub gp : f64,
}

impl EoSettings {
    ///
    /// Return new instance of EoSettings
    /// # Arguments 
    /// 
    /// * 'pop_size' : Number of search agents used by the learning algorithm.
    /// 
    /// * 'max_iter' : Maximum iteration to perform by the learning algorithm.
    /// 
    /// * 'lower_bound' : The minimum value can be set to a ANN weight or bias.
    /// 
    /// * 'lower_bound' : The maximum value can be set to a ANN weight or bias.
    /// 
    /// * 'a1' : EO parameter.
    /// 
    /// * 'a2' : EO parameter.
    /// 
    /// * 'gp' : EO parameter.
    ///  
    pub fn new(pop_size : usize, max_iter : usize, lower_bound : f64, upper_bound : f64,
        a1 : f64, a2 : f64, gp : f64)-> Self {        
        Self { 
            pop_size : usize::max(pop_size, 5),
            max_iter,
            lower_bound : f64::min(lower_bound, upper_bound),
            upper_bound : f64::max(lower_bound, upper_bound),
            a1,
            a2,
            gp,
        }
    }
}

impl Default for EoSettings{

     /// ## Return default parameters
    ///  
    /// * The population size (i.e., the count of search agents) = 50,
    /// 
    /// * The max iterations to perform by the training algorithm = 500,
    /// 
    /// * The lower bound for ANN weights and biases = -5.0, 
    /// 
    /// * The upper bound for ANN weights and biases = 5.0,
    /// 
    /// * a_1 = 2.0,
    /// 
    /// * a_2 = 1.0,
    /// 
    /// * gp = 0.5,
    /// 
    fn default() -> Self {
        Self{
            pop_size : 50,
            max_iter : 500,
            lower_bound : -5.0,
            upper_bound : 5.0,
            a1 : 2.0,
            a2 : 1.0,
            gp : 0.5,
        }
    }
}

impl Display for EoSettings{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "N: {}, Max_iter: {}, lb: {}, ub: {}, a1: {}, a2: {}, gp: {}", 
        self.pop_size,  self.max_iter, self.lower_bound, self.upper_bound, self.a1, self.a2, self.gp)
    }
}

/// Parameters for Growth Optimizer.
/// 
/// ## Example: 
/// 
/// let params : TrainerParams = TrainerParams::GoParams(50, 500, -5.0, 5.0);  
/// 
/// ### here,
///  
/// * The population size (i.e., the count of search agents) = 50.
/// 
/// * The max iterations to perform by the training algorithm = 500.
/// 
/// * The lower bound for ANN weights and biases = -5.0. 
/// 
/// * The upper bound for ANN weights and biases = 5.0.
  
#[derive(Debug, Clone, Copy)]
pub struct GoSettings{
    pub pop_size : usize,
    pub max_iter : usize,
    pub lower_bound : f64,
    pub upper_bound : f64,    
}
impl GoSettings{
    ///
    /// Return new instance of GoSettings
    /// 
    /// # Arguments 
    /// 
    /// * 'pop_size' : Number of search agents used by the learning algorithm.
    /// 
    /// * 'max_iter' : Maximum iteration to perform by the learning algorithm.
    /// 
    /// * 'lower_bound' : The minimum value can be set to a ANN weight or bias.
    /// 
    /// * 'lower_bound' : The maximum value can be set to a ANN weight or bias.
    /// 
    pub fn new(pop_size : usize, max_iter : usize, lower_bound : f64, upper_bound : f64)->Self{
        Self{
            pop_size : usize::max(pop_size, 5),
            max_iter,
            lower_bound : f64::min(lower_bound,upper_bound),
            upper_bound : f64::max(lower_bound,upper_bound) 
        }
    }
}

impl Default for GoSettings{
    fn default() -> Self {
        Self{
            pop_size : 50,
            max_iter : 500,
            lower_bound : -5.0,
            upper_bound : 5.0, 
        }
    }   
}

/// Parameters for Particle Swarm Optimizer (PSO).
    /// 
    /// /// ## Example: 
    /// 
    /// let params : TrainerParams = TrainerParams::PsoParams(50, 500, -5.0, 5.0, 2.0, 2.0);  
    /// 
    /// ### here,
    ///  
    /// * The population size (i.e., the count of search agents) = 50,
    /// 
    /// * The max iterations to perform by the training algorithm = 500,
    /// 
    /// * The lower bound for ANN weights and biases = -5.0, 
    /// 
    /// * The upper bound for ANN weights and biases = 5.0,
    /// 
    /// * c_1 = 2.0,
    /// 
    /// * c_2 = 2.0,
#[derive(Debug, Clone, Copy)]
    pub struct PsoSettings{
    pub pop_size : usize,
    pub max_iter : usize,
    pub lower_bound : f64,
    pub upper_bound : f64,
    pub c1 : f64,
    pub c2 : f64,
}
impl PsoSettings{
    ///
    /// Return new instance of PsoSettings
    /// 
    /// # Arguments 
    /// 
    /// * 'pop_size' : Number of search agents used by the learning algorithm.
    /// 
    /// * 'max_iter' : Maximum iteration to perform by the learning algorithm.
    /// 
    /// * 'lower_bound' : The minimum value can be set to a ANN weight or bias.
    /// 
    /// * 'lower_bound' : The maximum value can be set to a ANN weight or bias.
    /// 
    /// * 'c1' : PSO parameter.
    /// 
    /// * 'c2' : PSO parameter.
    /// 
    pub fn new(pop_size : usize, max_iter : usize, lower_bound : f64, upper_bound : f64, c1 : f64, c2 : f64)-> Self{        
        Self { 
            pop_size,
            max_iter,
            lower_bound,
            upper_bound,
            c1,
            c2,
        }
    }
}

impl Default for PsoSettings{
    fn default() -> Self {
        Self { 
            pop_size : 50,
            max_iter : 500,
            lower_bound : -5.0,
            upper_bound : 5.0,
            c1 : 2.0,
            c2 : 2.0,
        }
    }
}

//******************************************************************************************************

#[cfg(test)]
mod tests {
    //use super::*;
    //use linfa::dataset::DatasetView;
    //use ndarray::{Ix1, array};
    use super::{Layer, Evonet, Activations};

    #[test]
    fn test_ann_layer_when_size_null() {

        let mut layers : Vec<Layer> = Vec::new();
        layers.push(Layer::new(4, Activations::Sigmoid));
        layers.push(Layer::new(3, Activations::Sigmoid));
        layers.push(Layer::new(0, Activations::Linear));

        let ann = Evonet::new(layers);

        assert_eq!(ann.layers[2].neurons, 1);
    }

    #[test]
    fn test_update_neurons_when_null(){
        let mut l : Layer = Layer::new(10, Activations::SoftMax);
        l.update_neurons(0);

        assert_eq!(l.neurons, 1);
    }

    #[test]
    fn test_2_ann_layer_when_size_null() {

        let mut layers : Vec<Layer> = Vec::new();
        layers.push(Layer::new(0, Activations::Sigmoid));
        layers.push(Layer::new(3, Activations::Sigmoid));
        layers.push(Layer::new(0, Activations::Linear));

        let ann = Evonet::new(layers);

        assert_eq!(ann.layers[0].neurons, 1);
    }


    #[test]
    fn test_add_layer_fn() {
        let mut ann = Evonet::empty();
        ann.add_layer(Layer::new(4, Activations::Sigmoid));
        ann.add_layer(Layer::new(4, Activations::Sigmoid));
        ann.add_layer(Layer::new(4, Activations::Sigmoid));

        assert_eq!(ann.layers.len(), 3);
    }

    #[test]
    fn test_ann_update_layers() {
        let mut layers : Vec<Layer> = Vec::new();
        layers.push(Layer::new(4, Activations::Sigmoid));
        layers.push(Layer::new(3, Activations::Sigmoid));
        layers.push(Layer::new(0, Activations::Linear));

        let mut ann = Evonet::new(layers);

        ann.add_layer(Layer::new(5, Activations::ReLU));
        ann.add_layer(Layer::new(1, Activations::SoftMax));

        assert_eq!(ann.neuralnetwork.layers.len(), 5);
    }

    #[test]
    fn test_ann_clear_layers() {
        let mut layers : Vec<Layer> = Vec::new();
        layers.push(Layer::new(4, Activations::Sigmoid));
        let mut ann = Evonet::new(layers);

        ann.add_layer(Layer::new(5, Activations::ReLU));
        ann.add_layer(Layer::new(1, Activations::SoftMax));

        ann.clear_layers();

        assert_eq!(ann.neuralnetwork.layers.len(), 0);
    }
}