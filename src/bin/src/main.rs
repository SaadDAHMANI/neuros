extern crate neuros;
extern crate linfa;
extern crate ndarray;
extern crate sefar;

use ndarray::{array, Ix2};
use linfa::dataset::Dataset;

//--------------------------------------------------------------------------
use neuros::{activations::Activations, trainer::{Evonet, Layer, TrainerParams}};
//--------------------------------------------------------------------------
fn main() {
    println!("Hello, NEUROS!");
    
    ann_test_xor();
}

#[allow(dead_code)]
fn ann_test_xor(){

     // Give the ANN structure. 
     let mut layers_struct : Vec<Layer> = Vec::new();
     layers_struct.push(Layer::new(2, Activations::Sigmoid));
     layers_struct.push(Layer::new(4, Activations::Sigmoid));
     layers_struct.push(Layer::new(1, Activations::Linear));

    //Give input data samples
     let records = array![[0.,0.], [1.,0.], [0.,1.], [1.,1.]];
    
     //Give output data samples
     let targets = array![[0.], [1.], [1.], [0.]];
     
     //Create a data set from (inputs, outputs) samples
     let dataset : Dataset<f64, f64, Ix2> = Dataset::new(records, targets);


     // To use Growth optimizer (GO) as ANN trainer:
     let params : TrainerParams = TrainerParams::GoParams(50, 500, -5.0, 5.0);

     // To use Equilibrium optimizer (EO) as ANN trainer:
     //let params : TrainerParams = TrainerParams::EoParams(50, 500, -5.0, 5.0, 2.0, 1.0, 0.5);
     
     // To use Particle Swarm optimizer (PSO) as ANN trainer:
     //let params : TrainerParams = TrainerParams::PsoParams(50, 500, -5.0, 5.0, 2.0, 2.0);
  
     //Create an artificial neural network using the given .
    let mut ann = Evonet::new(layers_struct);

    // You can use 'add_layer()' function to add layer. 
    ann.add_layer(Layer::new(1, Activations::Linear));

    let train_result = ann.do_learning(&params, &dataset);

    println!("The ANN train results = {}", train_result.to_string());

    let sample = [0.0, 0.0];
    let resp = ann.compute_output(&sample);   

    println!("ANN response of :{:?} is {:?}", sample, resp);
 
}