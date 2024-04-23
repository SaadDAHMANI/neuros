extern crate neuros;
extern crate linfa;
extern crate ndarray;
extern crate sefar;

use ndarray::{array, Ix2};
use linfa::dataset::Dataset;

mod sin_example;
mod cos_example;


//------------------------------------------------------------------------------------------
use neuros::{activations::Activations, trainer::{self, EoSettings, Evonet, Layer, TrainingAlgo}};

use crate::{cos_example::ann_cos_test, sin_example::ann_sin_test};
//------------------------------------------------------------------------------------------
fn main() {
    println!("Hello, NEUROS!");
    
    //ann_xor_test();

    ann_sin_test();

    //ann_cos_test();
}

#[allow(dead_code)]
fn ann_xor_test(){

    //Give input data samples
    let records = array![[0.,0.], [1.,0.], [0.,1.], [1.,1.]];
    
    //Give output data samples
    let targets = array![[0.], [1.], [1.], [0.]];
     
    //Create a data set from (inputs, outputs) samples
    let dataset : Dataset<f64, f64, Ix2> = Dataset::new(records, targets);
    
     // Give the ANN structure. 
     let mut layers_struct : Vec<Layer> = Vec::new();
     layers_struct.push(Layer::new(dataset.records.dim().1, Activations::Sigmoid)); // 2 neurons
     layers_struct.push(Layer::new(4, Activations::Sigmoid));
     layers_struct.push(Layer::new(dataset.targets.dim().1, Activations::Linear)); // 1 neurons


    // To use Growth optimizer (GO) as ANN trainer:
    //let params : trainer::GoSettings =  trainer::GoSettings::new(50,500, -5.0, 5.0);
    // You can use default parameters,
    let params : trainer::GoSettings = trainer::GoSettings::default();
    let trainer : TrainingAlgo = TrainingAlgo::GO(params);
   
    // To use Equilibrium optimizer (EO) as ANN trainer:
    //let params : EoSettings = EoSettings::new(50,500, -10.0, 10.0, 2.0, 1.0, 0.5);
    // You can use default parameters,
    //let params : EoSettings = EoSettings::default();
    //let trainer : TrainingAlgo = TrainingAlgo::EO(params);
   
    // To use Particle Swarm optimizer (PSO) as ANN trainer:
    // let params : trainer::PsoSettings = trainer::PsoSettings::new(50, 500, -10.0, 10.0, 2.0, 2.0);
    // You can use default parameters,
    // let params : trainer::PsoSettings = trainer::PsoSettings::default();
    //let trainer : TrainingAlgo = TrainingAlgo::PSO(params);
             
     //Create an artificial neural network using the given .
    let mut ann = Evonet::new(layers_struct);

    // You can use 'add_layer()' function to add layer. 
    ann.add_layer(Layer::new(1, Activations::Linear));

    let train_result = ann.do_learning(&trainer, &dataset);

    println!("The ANN train results = {}", train_result.to_string());

    let sample = [0.0, 0.0];
    let resp = ann.compute_output(&sample);   

    println!("ANN response of :{:?} is {:?}", sample, resp);
 
}
