
use super::activations::*;
use super::neuralnet::*;

extern crate linfa;
//use linfa::dataset::Records;
use linfa::Dataset;

use ndarray::Ix2;

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
}

///
/// Parameters of the training algorithm.
/// # Arguments 
/// 
#[derive(Debug, Clone)]
pub enum TrainerParams{

    /// Parameters for Equilibrium Optimizer (EOs).
    /// 
    /// ## Example: 
    /// 
    /// let params : TrainerParams = TrainerParams::EoParams(50, 500, -5.0, 5.0, 2.0, 1.0, 0.5);  
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
    EoParams(usize, usize, f64, f64, f64, f64, f64),

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
    PsoParams(usize, usize, f64, f64, f64, f64),
    
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
    GoParams(usize, usize, f64, f64),
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
    /// Create an empty neural network
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
    /// add a single layer to the neural network.
    ///   
    pub fn add_layer(&mut self, layer : Layer){       
        self.layers.push(layer);
        self.update_neuralnet();       
    }

    ///
    /// Perform ANN training
    ///  
    pub fn do_learning(&mut self, params : &TrainerParams, train_dataset : &'a Dataset<f64, f64, Ix2>)-> OptimizationResult {
        self.learning_set = Some(train_dataset);
        let wb = self.neuralnetwork.get_weights_biases_count();
                
        let result =  match params{
            // Use EO as trainer
            TrainerParams::EoParams(p,k, lbv, ubv, a1, a2, gp) => {
                let lb = vec![*lbv; wb];
                let ub = vec![*ubv; wb];
                let mut settings : EOparams = EOparams::new(*p, wb, *k, &lb, &ub, *a1, *a2, *gp).unwrap();
                settings.dimensions = wb;
                let mut algo = EO::<Evonet>::new(&settings, self);
                 algo.run()               
            },

            // Use POS as trainer
            TrainerParams::PsoParams(p,k, lbv, ubv, c1, c2)=>{
                
                let lb = vec![*lbv; wb];
                let ub = vec![*ubv; wb];
                let mut settings : PSOparams = PSOparams::new(*p, wb, *k, &lb, &ub, *c1, *c2).unwrap();
                settings.dimensions = wb;
                let mut algo = PSO::<Evonet>::new(&settings, self);
                algo.run()
            },

            // Use GO as trainer
            TrainerParams::GoParams(p,k, lbv, ubv) => {

                let lb = vec![*lbv; wb];
                let ub = vec![*ubv; wb];
                let mut settings : GOparams = GOparams::new(*p, wb, *k, &lb, &ub);

                settings.dimensions = wb;
                let mut algo = GO::<Evonet>::new(&settings, self);
                algo.run()
                 
            },
        };       
                     
        //
        result

    }

    pub fn compute_outputs(&mut self, _dataset : &Dataset<f64, f64, Ix2>)-> Result<Vec<Vec<f64>>, String>{
         Err(String::from("Not implemented yet!"))
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
                let target_features = learning_set.targets.dim().1;
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
                                        for j in 0..target_features{
                                            errors[j] += (y_vector[j] - computed[j]).powi(2);
                                        }                            
                                   },
                            };
                        },
                    }
                };

                // compute RMSE for learning samples for each output
                for i in 0..target_features {
                    errors[i] = f64::sqrt(errors[i]/target_features as f64);
                }              
        
                //learning_err = sum of RMSE errors:
                let rmse_error : f64 = errors.iter().fold(0.0f64, |sum, a| sum + a);
                rmse_error
            },
        }   
    }
}



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