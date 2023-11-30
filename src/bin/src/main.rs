use neuros::{trainer::Evonet, activations::Activations};

extern crate neuros;
use linfa::dataset::DatasetView;
use ndarray::{Ix1, array};


fn main() {
    println!("Hello, world!");
    let layers = [2, 3, 1].to_vec();
    let activations = [Activations::Sigmoid, Activations::Sigmoid, Activations::Sigmoid].to_vec();

    let records = array![[1.,1.], [2.,1.], [3.,2.], [4.,1.],[5., 3.], [6.,2.]];
    let targets = array![1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    
    let dataset : DatasetView<f64, f64, Ix1> = (records.view(), targets.view()).into();

    let k_fold : Option<usize> = Some(2);

    let mut ann = Evonet::new(&layers, &activations, &dataset.to_owned(), k_fold);


}
