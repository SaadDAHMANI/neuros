pub mod activations;
pub mod neuralnet;
pub mod trainer;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("The lower bound of search space {} can not be greater than the upper bound {}!", lb, ub)]
    InvalidSearchBounds{lb : f64, ub : f64},

    #[error("The number of search agents (population size) can not be too small!")]
    SearchPopulationTooSmall{n: usize},

    #[error("Error in ANN input length. Expected: {}, Found: {}", expected, actual)]
    InvalidInputCount{expected : usize, actual : usize},

    #[error("Ann error is occured : {}", msg)]
    Other{msg: String},
}
/* impl Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error : {:?}", self)
    }
} 
 */
