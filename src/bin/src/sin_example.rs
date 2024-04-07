// This code demonstrates how to utilize the Neuros crate to predict the output 
// of the function f(x,y) = sin(x*y).

#[allow(dead_code)]
fn ann_test_sin(){

    let path = "src/bin/src/data/sin_xy.csv";
    //--------------------------
    let data = read_csv_file(path);

    match data{
        Err(eror) => println!("I can not read the given file because of :{:?}", eror),
        Ok(data)=> {
                        
            // Split the data to take the first two columns as inputs and the third (last) column as output.
            // records : inputs,
            // targets : outputs.
            let (records, targets) = data.view().split_at(Axis(1), 2);
            
            ////Create a data set from (inputs, outputs) samples
            let dataset : Dataset<f64, f64, Ix2> = Dataset::new(records.to_owned(),targets.to_owned());
           
            // shuffle the dataset
            let shuffle : bool = true;
        
            //split the dataset into (80% learning, 20% testing) 
            let split_ratio : f32 = 0.8;
           
            // Give the ANN structure. 
            let layers = [2, 3, 1].to_vec();

            // //Give activation function for each layer.
            let activations = [Activations::Sigmoid, Activations::Sigmoid, Activations::Linear].to_vec();
 
            //Create an artificial neural network using the given parameters.
            let mut ann_restult = Evonet::new(&layers, &activations, &dataset, shuffle, split_ratio);
            
            match &mut ann_restult{
                Err(error) => panic!("Finish due to error : {}", error),
                Ok(ann)=>{
                    // define parameters for the training (learning algorithm) 
                    let population_size : usize = 50; // set the search poplation size,
                    let dimensions: usize = ann.get_weights_biases_count(); // get the search space dimension, 
                    let max_iterations : usize = 500; // set the maximum number of iterations (learning step),
                    let lb = vec![-5.0; dimensions]; // set the lower bound for the search space,
                    let ub = vec![5.0; dimensions]; // set the upper bound for the search space,

                    // create the GO parameters (+learning algorithm)
                    let params : GOparams = GOparams {
                        population_size,
                        dimensions,
                        max_iterations,
                        lower_bounds: &lb, 
                        upper_bounds : &ub,
                    };  

                    let trainer_params = TrainerParams::GoParams(params); 
    
                    // perform the learning step. 
                    let learning_results = ann.do_learning(&trainer_params);

                    println!("Growth Optimizer trainer. Prediction of f(x,y) = sin(x*y) : RMSE_Learning = {:?}", learning_results.best_fitness);                    
                },
            };
        },
    };
}

///
/// Read data from CSV file
/// path : csv file path 
/// 
fn read_csv_file(path : &str)-> Result<Array2<f64>, Box<dyn Error>>{
    // Read an array back from the file
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    //let array_read: Array2<u64> = reader.deserialize_array2((2, 3))?;
    let array_read: Array2<f64> = reader.deserialize_array2_dynamic()?;

    Ok(array_read)
}
