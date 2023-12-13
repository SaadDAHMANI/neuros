pub mod activations;
pub mod neuralnet;
pub mod trainer;




#[cfg(test)]
mod tests {
    use super::*;
    use linfa::dataset::DatasetView;
    use ndarray::{Ix1, array};

    #[test]
    fn it_works() {
        let result = 2+2;
        assert_eq!(result, 4);
    }
    
    #[test]
    fn test_dataset_records(){
        let records = array![[1.,1.], [2.,1.], [3.,2.]];
        let targets = array![1.0, 1.5, 2.5];

        let dataset : DatasetView<f64, f64, Ix1> = (records.view(), targets.view()).into();
        
        println!("dataset = \n {:?}", dataset);

        assert_eq!(1+1, 2);

    }
    
}


