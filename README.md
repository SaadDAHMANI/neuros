# Neuros

[Neuros](https://github.com/SaadDAHMANI/neuros) is a [Rust](https://www.rust-lang.org/) package for Artificial (Feedforward) Neural Networks (ANNs) processing. [Neuros](https://github.com/SaadDAHMANI/neuros) uses [Sefar](https://crates.io/crates/sefar) crate to perform ANNs training. 

In the learning (training) stage, [Neuros](https://github.com/SaadDAHMANI/neuros) minimizes the Root Mean Square Error (RMSE) between computed and given model outputs. 

$Error_{Learning} = RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ 

Where $y_i$ and $\hat{y}_i$ represent the computed and given model outputs, respectively.

In the case of multiple ANN outputs, [Neuros](https://github.com/SaadDAHMANI/neuros) minimizes the sum of RMSE.  

$Error_{Learning} = \sum RMSE$ 

## Training algorithms 
The current version supports the following training algorithms:

[x] Equilibrium Optimizer (EO), 

[x] Particle Swarm Optimizer (PSO),

[x] Growth optimizer (GO).

## How to use Neuros
Please, **check the folder [src/bin/src](https://github.com/SaadDAHMANI/neuros/tree/master/src/bin/src) for the examples**.

### Example (ANN with single output)

1. Import dependencies:

```toml
[dependencies]
neuros = "0.1.1"
ndarray ="0.15.6"
linfa ="0.7.0"
```
2. In the main.rs file: 

```rust

```

