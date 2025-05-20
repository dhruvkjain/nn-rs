# Generalized and Modularized Neural Network from scratch in Rust

just specify layers and activation functions, loss function and optimizer and have fun 😉

## 🛠️ Usage
```rust
fn main() -> Result<(), Box<dyn Error>>{
    let (x, y) = load_training_data()?;

    let mut trainer = NN {
        layers: vec![
            Box::new(Layer::new(784, 128)),
            Box::new(ReLu::new()),
            Box::new(Layer::new(128, 10)),
        ],
        loss_fn: MSELoss { targets: None },
        optim: SGDOptimizer { lr: 0.01 },
    };
    
    for iteration in 0..251 {
        let loss = trainer.train_step(&x, &y);
        
        if iteration%50 == 0{
            println!("Iteration {}: loss = {}", iteration, loss);
        }
    }
    Ok(())
}
```


## 🔧 Features
- **Layers can be composed and stacked flexibly in any order**
- Support for trainable layers (`Layer`) and non-trainable layers/activation functional layer (`ReLU`, `Sigmoid`)
- Custom `Propagate` trait with forward and backward passes
- `Loss` trait with MSE loss implementation
- Basic SGD optimizer
- Composable layer structure
- Written purely in safe Rust


## 🧩 Crate Structure
```shell
nn-rs/
├── Cargo.toml                ← workspace root
├── crates/
│   ├── layers/               # Contains Propagate trait, Layer, and activation functions
│   ├── loss/                 # Contains Loss trait and implementations like MSELoss
│   ├── optimizer/            # Contains Optimizer trait and implementations like SGDOptimizer
│   └── model/                # Assembles layers, loss and optimizer to handles training
└── examples/
    ├── test_data
    │    └── mnist/
    │       └── mnist_train.csv
    └── mnist-nn/
        ├── Cargo.toml        ← depends on crates via path
        └── src/
            └── main.rs       # Import the model crate here, specify layers and activation functions, loss function and optimizer
```


## Results for MNIST dataset
1 hidden layer of 128 neuron, relu activation function, 250 iterations
```shell
DATA: 60000, 784
LABELS: 60000, 1
Iteration 0: loss = 7000.565
Iteration 50: loss = 38.300846
Iteration 100: loss = 32.709435
Iteration 150: loss = 28.24366
Iteration 200: loss = 24.61995
Iteration 250: loss = 21.667906
```


1 hidden layer of 128 neuron, relu activation function, 15 iterations
```shell
DATA: 60000, 784
LABELS: 60000, 1
Iteration 0: loss = 4730.49
Iteration 5: loss = 34.387665
Iteration 10: loss = 33.08763
Iteration 15: loss = 32.200054
```
