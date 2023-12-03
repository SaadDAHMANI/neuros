use rand::Rng;
use super::activations::*;

#[derive(Debug, Clone)]
pub struct Neuralnet {
	pub neurons:Vec<Vec<f64>>,
	pub weights:Vec<Vec<Vec<f64>>>,
	pub biases:Vec<Vec<f64>>,
	pub activations:Vec<Activations>,
	pub layers:Vec<usize>,
	//pub learning_rate: f64,
	//pub cost: f64

	//----------------------------
}

#[allow(dead_code)]
impl Neuralnet {

	pub fn new(layers:Vec<usize>, activations:Vec<Activations>)-> Neuralnet
	{
		let mut nn = Neuralnet {
			neurons: Vec::new(),
			weights: Vec::new(),
			biases: Vec::new(),
			activations,
			layers,			
			//learning_rate: 0.01,
			//cost: 0.0
		};

		nn.init_neurons();
		nn.init_biases();
		nn.init_weights();

		return nn;

	}

	pub fn init_neurons(&mut self)
	{
		self.neurons = Vec::new();
		for x in 0..self.layers.len() {
			let neurons = vec![0.0; self.layers[x]];
			self.neurons.push(neurons);
		}
	}

	pub fn init_biases(&mut self)
	{
		let mut rng = rand::thread_rng();
		self.biases = Vec::with_capacity(self.layers.len());

		for i in 1..self.layers.len() {
			let num_items = self.layers[i];
			let mut bias : Vec<f64> = vec![0.0; num_items];
			for j in 0..num_items {
				bias[j] = rng.gen_range(-0.5..0.5) / num_items as f64;
			}
			self.biases.push(bias);
		}
	}

	pub fn init_weights(&mut self)
	{
		let mut rng = rand::thread_rng();
		self.weights = Vec::new();

		for i in 1..self.layers.len() {
			let num_prev_items = self.layers[i-1];
			let mut layer_weights : Vec<Vec<f64>> = Vec::new();
			for _j in 0..self.layers[i] {
				let mut neuron_weights : Vec<f64> = vec![0.0; num_prev_items];
				for k in 0..num_prev_items {
					neuron_weights[k] = rng.gen_range(-0.5..0.5) / num_prev_items as f64;
				}
				layer_weights.push(neuron_weights);
			}
			self.weights.push(layer_weights);
		}
	}

	pub fn activate(&self, x:f64, layer_id: usize) -> f64 {
		match self.activations[layer_id] {
			Activations::Sigmoid => sigmoid(x),
			Activations::ReLU => relu(x),
			Activations::LeakyRelu => leakyrelu(x),
			Activations::TanH => tanh(x),
			Activations::SoftMax => softmax(x),
			_ => x
		}
	}

	pub fn feed_forward(&mut self, inputs:&[f64]) -> Vec<f64> {
		if inputs.len() != self.layers[0] {
			panic!("Inputs [{}] length is not equal to neuralnet first layer dimension [{}].", inputs.len(), self.layers[0]);
		} 

		for i in 0..inputs.len() {
			self.neurons[0][i] = inputs[i];
		}
		
		//for i in 1..inputs.len() {
		for i in 1..self.layers.len() {
			
			let layer_idx = i - 1;

			for j in 0..self.layers[i] {
				let mut value:f64 = 0.0;
				for k in 0..self.layers[i-1] {
					value += self.weights[i - 1][j][k] * self.neurons[i - 1][k];
				}
				self.neurons[i][j] = self.activate(value + self.biases[i - 1][j], layer_idx);
			}

			match self.activations[layer_idx] {
				Activations::SoftMax => {
					let mut sigma : f64 = 0.0;
					for j in 0..self.layers[i] {
						sigma += self.neurons[i][j];
					}
					for j in 0..self.layers[i] {
						self.neurons[i][j] /= sigma;
					}
				},
				_ => {}
			}
		}
		self.neurons[self.layers.len() - 1].clone()
	}                                                                            

	 fn update_weights(&mut self, new_weights : &Vec<f64>) {

		let mut l : usize = 0;

		let number_of_layers = self.layers.len();
		
		for i in 1..number_of_layers {

			let neurons_previous_layer = self.layers[i-1];

			for j in 0..self.layers[i] {

				for k in 0..neurons_previous_layer {

					self.weights[i-1][j][k] = new_weights[l];					
					l+=1;
				}
			}
		}
	}

	fn update_biases(&mut self, new_biases : &Vec<f64>){
		
		let mut l : usize =0;

		for i in 1..self.layers.len() {
			
			let num_nodes = self.layers[i];

			for j in 0..num_nodes {

				self.biases[i-1][j] = new_biases[l];
				
				l+=1;
			}

		}
	}

	pub fn update_weights_biases(&mut self, new_weights_biases : &Vec<f64> ){

		let mut l : usize = 0;

		let number_of_layers = self.layers.len();
		
		for i in 1..number_of_layers {

			let neurons_previous_layer = self.layers[i-1];

			for j in 0..self.layers[i] {

				for k in 0..neurons_previous_layer {

					self.weights[i-1][j][k] = new_weights_biases[l];					
					l+=1;
				}
			}
		}

		for i in 1..self.layers.len() {
			
			let num_nodes = self.layers[i];

			for j in 0..num_nodes {

				self.biases[i-1][j] = new_weights_biases[l]; //1.0;
				
				l+=1;
			}

		}
	}

    fn get_weights_count(&self)-> usize {

		let mut count : usize =0;
		
		for i in 1..self.layers.len() {
		   count += self.layers[i-1]* self.layers[i]; 
		}
		count
	   }
   
    fn get_biases_count(&self)->usize {
		
		   let mut count : usize =0;
		
		   for i in 1..self.layers.len() {
				count += self.layers[i]; 
		   }
		   count
	   }
   
    pub fn get_weights_biases_count(&self)-> usize {
		  self.get_weights_count()+ self.get_biases_count()
	}
	///
	/// Compute error using RMSE formula
	///
	pub fn compute_learning_error_rmse(&mut self, learn_in : &Vec<Vec<f64>>,  expected_learn_out : &Vec<Vec<f64>>)->f64 {

        let mut totalerror : f64 = 0.0f64;
        let mut err : f64 =0.0f64;

		let incount = learn_in.len();

		//println!("Wi = {:?}", self.weights);

        for i in 0..incount {

            let computed = self.feed_forward(&learn_in[i]);

            for j in 0..computed.len(){
               err = f64::powi(expected_learn_out[i][j]- computed[j], 2); 
			   //println!("err : {}", err);
            }
            totalerror +=err;
        }
		//println!("err : {}", totalerror);
		
        totalerror = f64::sqrt(totalerror/incount as f64);
		return totalerror;
    } 
	
   }