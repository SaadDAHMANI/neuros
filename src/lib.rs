use std::fmt::Display;

pub mod activations;
pub mod neuralnet;
pub mod trainer;

#[derive(Debug, Clone)]
pub enum Error {  
    InvalidSearchBounds{lb : f64, ub : f64},
    InvalidInputCount{expected : usize, actual : usize},
    Other{msg: String},
}

impl std::fmt::Display for crate::Error{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidInputCount {expected, actual} => write!(f, "Error in input count, expected: {}, found: {} !", expected, actual),
            Error::InvalidSearchBounds {lb, ub }=> write!(f, "Search lower bound {}, can not be greater than upper bound {}", lb, ub), 
            Error::Other {msg } => write!(f, "Error: {}", msg),
        }      
    }
}