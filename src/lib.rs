pub mod activations;
pub mod neuralnet;
pub mod trainer;

#[cfg(test)]
mod tests {
    //use super::*;
    //use linfa::dataset::DatasetView;
    //use ndarray::{Ix1, array};

    #[test]
    fn it_works() {
        let result = 2+2;
        assert_eq!(result, 4);
    }
}


