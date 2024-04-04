// let targets: ArrayBase<ViewRepr<&f64>, Dim<[usize; 2]>>
// convert : ArrayBase<ViewRepr<&f64>, Dim<[usize; 2]>> to ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>


///
/// Run training using Equilibrium Optimizer.
///   
fn test_eo_trainer(ann : &mut Evonet){
    // define arameters for the training (learning algorithm) 
    let population_size : usize = 100; // set the search poplation size,
    let dimensions: usize = ann.get_weights_biases_count(); // get the search space dimension, 
    let max_iterations : usize = 1000; // set the maximum number of iterations (learning step),
    let lb = vec![-5.0; dimensions]; // set the lower bound for the search space,
    let ub = vec![5.0; dimensions]; // set the upper bound for the search space,
    let a1 : f64 = 2.0; // give the value of a1 parameter (Equilibrium Optimizer),
    let a2 : f64 = 2.0; // give the value of a2 parameter (Equilibrium Optimizer),
    let gp :f64 = 0.5; // give the value of GP parameter (Equilibrium Optimizer),

    // create the EO parameters (learning algorithm)
    let params : EOparams = EOparams::new(population_size, dimensions, max_iterations, &lb, &ub, a1, a2, gp);  
    let trainer_params = TrainerParams::EoParams(params); 
    
    // perform the learning step. 
   let learning_results = ann.do_learning(&trainer_params);

   println!("RMSE_Learning = {:?}", learning_results.best_fitness);
}

///
/// Run training using Equilibrium Optimizer.
///   
fn test_pso_trainer(ann : &mut Evonet){
    // define arameters for the training (learning algorithm) 
    let population_size : usize = 100; // set the search poplation size,
    let dimensions: usize = ann.get_weights_biases_count(); // get the search space dimension, 
    let max_iterations : usize = 1000; // set the maximum number of iterations (learning step),
    let lb = vec![-5.0; dimensions]; // set the lower bound for the search space,
    let ub = vec![5.0; dimensions]; // set the upper bound for the search space,
    let c1 : f64 = 2.0; // give the value of a1 parameter (EPSO),
    let c2 : f64 = 2.0; // give the value of a2 parameter (PSO),
    
    // create the EO parameters (learning algorithm)
    let params : PSOparams = PSOparams::new(population_size, dimensions, max_iterations, &lb, &ub, c1, c2);  
    let trainer_params = TrainerParams::PsoParams(params); 
    
    // perform the learning step. 
   let learning_results = ann.do_learning(&trainer_params);

   println!("RMSE_Learning = {:?}", learning_results.best_fitness);
}
