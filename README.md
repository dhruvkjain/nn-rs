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
- Support for trainable layers (`Layer`) and non-trainable layers/activation functional layer (`ReLU`, `LeakyRelU`, `ELU`, `SELU`).
- He, LeCun, Glorot Intiazliations for weights and biasis.
- Custom `Propagate` trait with forward and backward passes.
- `Loss` trait with MSE loss, Cross Entropy Loss implementations
- `Optimizer` trait with SGD optimizer, Momentum Optimizer, RMSProp Optimizer, Nesterov Accelerated Gradient, Adam, Nadam implementations.
- Composable layer structure
- Written purely in safe Rust


## 🧩 Crate Structure
```shell
nn-rs/
├── Cargo.toml                ← workspace root
├── crates/
│   ├── layers/               # Contains Propagate trait, Layer, and activation functions
│   ├── loss/                 # Contains Loss trait and implementations like MSELoss, CrossEntropyLoss
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

## Performance Results
- **RMSProp ~ Nadam > AdaMax > Adam > Nestrov > Momentum > SGD > GD** 
- **SELU > ELU > leaky ReLU (and its variants) > ReLU > tanh > logistic**
- To learn more about these functions read my research work [here](https://dhruvkjain.github.io/pkms/ML/Generalized-NN-in-Rust). 

## Results for MNIST dataset

#### 1 hidden layer of 16 neuron He intialized, elu activation function, cross entropy loss, nadam optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.24601953
Iteration 10: loss = 0.061340604
Iteration 20: loss = 0.04186616
Iteration 30: loss = 0.035088763
Iteration 40: loss = 0.03133954
Iteration 50: loss = 0.02880185
Iteration 60: loss = 0.026939582
Iteration 70: loss = 0.025421605
Iteration 80: loss = 0.024120415
Iteration 90: loss = 0.022980707
Iteration 100: loss = 0.021966562
Iteration 110: loss = 0.021057272
Iteration 120: loss = 0.020238798
Iteration 130: loss = 0.019499602
Iteration 140: loss = 0.018829359
Iteration 150: loss = 0.018218907

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1

Confusion Matrix:
[[964, 0, 1, 2, 1, 5, 4, 1, 2, 0],
 [0, 1115, 4, 1, 0, 1, 3, 2, 9, 0],
 [6, 7, 966, 12, 8, 1, 7, 5, 20, 0],
 [3, 1, 12, 941, 0, 19, 0, 13, 18, 3],
 [1, 1, 4, 1, 939, 0, 9, 2, 3, 22],
 [5, 2, 1, 21, 9, 822, 13, 0, 13, 6],
 [9, 3, 3, 0, 6, 14, 919, 0, 4, 0],
 [1, 10, 24, 3, 5, 1, 0, 953, 2, 29],
 [10, 3, 5, 20, 8, 14, 8, 11, 890, 5],
 [8, 4, 2, 11, 27, 8, 0, 11, 4, 934]]

Class 0: TP = 964, FP = 43, FN = 16, TN = 8977
Precision = 0.95729893
Recall(Sensitivity) = 0.98367345
F1-score: = 0.970307
---------------------------------------------
Class 1: TP = 1115, FP = 31, FN = 20, TN = 8834
Precision = 0.9729494
Recall(Sensitivity) = 0.98237884
F1-score: = 0.97764134
---------------------------------------------
Class 2: TP = 966, FP = 56, FN = 66, TN = 8912
Precision = 0.94520545
Recall(Sensitivity) = 0.93604654
F1-score: = 0.9406037
---------------------------------------------
Class 3: TP = 941, FP = 71, FN = 69, TN = 8919
Precision = 0.9298419
Recall(Sensitivity) = 0.9316832
F1-score: = 0.93076164
---------------------------------------------
Class 4: TP = 939, FP = 64, FN = 43, TN = 8954
Precision = 0.93619144
Recall(Sensitivity) = 0.9562118
F1-score: = 0.94609576
---------------------------------------------
Class 5: TP = 822, FP = 63, FN = 70, TN = 9045
Precision = 0.9288136
Recall(Sensitivity) = 0.92152464
F1-score: = 0.92515475
---------------------------------------------
Class 6: TP = 919, FP = 44, FN = 39, TN = 8998
Precision = 0.95430946
Recall(Sensitivity) = 0.9592902
F1-score: = 0.9567933
---------------------------------------------
Class 7: TP = 953, FP = 45, FN = 75, TN = 8927
Precision = 0.9549098
Recall(Sensitivity) = 0.9270428
F1-score: = 0.94077
---------------------------------------------
Class 8: TP = 890, FP = 75, FN = 84, TN = 8951
Precision = 0.9222798
Recall(Sensitivity) = 0.9137577
F1-score: = 0.91799897
---------------------------------------------
Class 9: TP = 934, FP = 65, FN = 75, TN = 8926
Precision = 0.9349349
Recall(Sensitivity) = 0.92566895
F1-score: = 0.9302789
---------------------------------------------
Loss: 0.019970542
Accuracy: 94.43%
```


#### 1 hidden layer of 16 neuron, leaky relu activation function, cross entropy loss, momentum optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.3874951
Iteration 50: loss = 0.08339318
Iteration 100: loss = 0.052310627
Iteration 150: loss = 0.04365603

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.042081427
Accuracy: 87.29%
```

#### 1 hidden layer of 16 neuron, elu activation function, cross entropy loss, rmsprop optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.34491262
Iteration 50: loss = 0.037369683
Iteration 100: loss = 0.027343063
Iteration 150: loss = 0.021294897

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.021050507
Accuracy: 93.72%
```

#### 1 hidden layer of 16 neuron, selu activation function, cross entropy loss, rmsprop optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.37218064
Iteration 50: loss = 0.038395952
Iteration 100: loss = 0.031207351
Iteration 150: loss = 0.02991341

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.026730014
Accuracy: 91.59%
```

#### 1 hidden layer of 16 neuron, elu activation function, cross entropy loss, adam optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.3719147
Iteration 50: loss = 0.04161721
Iteration 100: loss = 0.029627483
Iteration 150: loss = 0.02466298

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.025647987
Accuracy: 92.46%
```

#### 1 hidden layer of 16 neuron, elu activation function, cross entropy loss, nadam optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.3536123
Iteration 50: loss = 0.035201367
Iteration 100: loss = 0.025516039
Iteration 150: loss = 0.021433324

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.02411928
Accuracy: 92.94%
```

#### 1 hidden layer of 16 neuron He intialized, elu activation function, cross entropy loss, nadam optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.24853164
Iteration 50: loss = 0.030401217
Iteration 100: loss = 0.02370399
Iteration 150: loss = 0.019435808

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.020764414
Accuracy: 94%
```