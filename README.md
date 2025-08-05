# Generalized and Modularized Neural Network from scratch in Rust

just specify layers and activation functions, loss function and optimizer and have fun 😉

## 🛠️ Usage
```rust
// here x is of [batch size, number of classes]
let (x, y) = load_training_data()?;

// ----------------------MODEL-----------------------
let mut nn = NN {
    layers: vec![
        LayerTypes::Layer(Layer::new(784, 64, Initialization::He, Regularization::ElasticNet { l1: 0.0001, l2: 0.0001 })),
        LayerTypes::ELU(ELU::new(1.0)),
        LayerTypes::Layer(Layer::new(64, 32, Initialization::He, Regularization::ElasticNet { l1: 0.0001, l2: 0.0001 })),
        LayerTypes::ELU(ELU::new(1.0)),
        LayerTypes::Layer(Layer::new(32, 16, Initialization::He, Regularization::ElasticNet { l1: 0.0001, l2: 0.0001 })),
        LayerTypes::ELU(ELU::new(1.0)),
        LayerTypes::Layer(Layer::new(16, 10, Initialization::He, Regularization::ElasticNet { l1: 0.0001, l2: 0.0001 })),
    ],
    loss_fn: CrossEntropyLoss { probs: None, one_hot_encoded:None },
    optim: NadamOptimizer { 
        lr: 0.0005, 
        momentum: 0.9,
        decay_rate: 0.999, 
        smoothing: 1e-7 as f32,
        velocity_w: Vec::new(),
        velocity_b: Vec::new(),
        scaling_factor_w: Vec::new(), 
        scaling_factor_b: Vec::new(),
        timestep: 0,
    },
    regularization: Regularization::ElasticNet { l1: 0.0001, l2: 0.0001 },
};

// ---------------------TRAINING----------------------
for iteration in 0..=150 {
    let (loss, accuracy) = nn.train_step(&x, &y, iteration, 150, "../test_data/mnist");
    
    if iteration%10 == 0 {
        println!("Iteration {}: loss = {}, accuracy = {}%", iteration, loss, accuracy);
    }
}

// ----------------TESTING & ACCURACY-----------------
let (xt, yt) = load_testing_data()?;
let num_layers = nn.layers
    .iter()
    .filter(|l| matches!(l, LayerTypes::Layer(_)))
    .count();
let (loaded_weights, loaded_bias) = load_model_weights_and_biases("../test_data/mnist", num_layers)?;

for (i, layer) in nn.layers.iter_mut().filter_map(|l| match l {
    LayerTypes::Layer(layer) => Some(layer),
    _ => None,
}).enumerate() {
    layer.set_params(loaded_weights[i].clone(), loaded_bias[i].clone());
}

let (loss, accuracy) = nn.test_step(&xt, &yt);
println!("Loss: {loss}");
println!("Accuracy: {accuracy}%");
```


## 🔧 Features
- **Layers can be composed and stacked flexibly in any order**
- Support for trainable layers (`Layer`) and non-trainable layers/activation functional layer (`ReLU`, `LeakyReLU`, `ELU`, `SELU`).
- `He`, `LeCun`, `Glorot` initializations for weights and biasis.
- `Lasso(L1)`, `Ridge(L2)`, `Elastic Net(L1+L2)` regularizations. 
- Custom `Propagate` trait with forward and backward passes.
- `Loss` trait with `MSE loss`, `Cross Entropy Loss` implementations
- `Optimizer` trait with `SGD optimizer`, `Momentum Optimizer`, `RMSProp Optimizer`, `Nesterov Accelerated Gradient`, `Adam`, `Nadam` implementations.
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
- To learn more about these functions read [this](https://dhruvkjain.github.io/pkms/ML/Generalized-NN-in-Rust). 

## Results for MNIST dataset

#### 784-64-32-16-10 architecture, He intialized, elastic net regularization, elu activation function, cross entropy loss, nadam optimizer, 0.0005 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.515229, accuracy = 7.3083334%
Iteration 10: loss = 0.4194208, accuracy = 50.795002%
Iteration 20: loss = 0.3690825, accuracy = 69.354996%
Iteration 30: loss = 0.33651012, accuracy = 77.49167%
Iteration 40: loss = 0.31396595, accuracy = 81.56%
Iteration 50: loss = 0.29716164, accuracy = 84.066666%
Iteration 60: loss = 0.28394672, accuracy = 85.78333%
Iteration 70: loss = 0.27313048, accuracy = 87.06833%
Iteration 80: loss = 0.26401412, accuracy = 88.08833%
Iteration 90: loss = 0.25617692, accuracy = 88.905%
Iteration 100: loss = 0.2493262, accuracy = 89.57333%
Iteration 110: loss = 0.24332188, accuracy = 90.18166%
Iteration 120: loss = 0.23802409, accuracy = 90.67333%
Iteration 130: loss = 0.23332518, accuracy = 91.03333%
Iteration 140: loss = 0.22913754, accuracy = 91.385%
Iteration 150: loss = 0.22536595, accuracy = 91.67%

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1

Confusion Matrix:
[[956, 0, 2, 2, 0, 6, 6, 2, 6, 0],
 [0, 1113, 2, 2, 0, 3, 4, 1, 10, 0],
 [10, 2, 930, 14, 10, 1, 15, 10, 35, 5],
 [7, 3, 22, 885, 2, 38, 2, 13, 32, 6],
 [1, 4, 6, 0, 903, 1, 18, 2, 4, 43],
 [9, 3, 4, 25, 3, 786, 12, 10, 35, 5],
 [12, 3, 7, 0, 15, 17, 901, 0, 3, 0],
 [1, 11, 23, 8, 8, 0, 0, 947, 1, 29],
 [10, 7, 11, 32, 7, 27, 13, 12, 848, 7],
 [11, 6, 1, 11, 33, 4, 2, 13, 9, 919]]
Class 0: TP = 956, FP = 61, FN = 24, TN = 8959
Precision = 0.94001967
Recall(Sensitivity) = 0.9755102
F1-score: = 0.95743614
---------------------------------------------
Class 1: TP = 1113, FP = 39, FN = 22, TN = 8826
Precision = 0.9661458
Recall(Sensitivity) = 0.98061675
F1-score: = 0.97332746
---------------------------------------------
Class 2: TP = 930, FP = 78, FN = 102, TN = 8890
Precision = 0.92261904
Recall(Sensitivity) = 0.9011628
F1-score: = 0.9117647
---------------------------------------------
Class 3: TP = 885, FP = 94, FN = 125, TN = 8896
Precision = 0.90398365
Recall(Sensitivity) = 0.87623763
F1-score: = 0.8898944
---------------------------------------------
Class 4: TP = 903, FP = 78, FN = 79, TN = 8940
Precision = 0.9204893
Recall(Sensitivity) = 0.9195519
F1-score: = 0.9200204
---------------------------------------------
Class 5: TP = 786, FP = 97, FN = 106, TN = 9011
Precision = 0.8901472
Recall(Sensitivity) = 0.8811659
F1-score: = 0.88563377
---------------------------------------------
Class 6: TP = 901, FP = 72, FN = 57, TN = 8970
Precision = 0.9260021
Recall(Sensitivity) = 0.94050103
F1-score: = 0.9331953
---------------------------------------------
Class 7: TP = 947, FP = 63, FN = 81, TN = 8909
Precision = 0.93762374
Recall(Sensitivity) = 0.92120624
F1-score: = 0.9293425
---------------------------------------------
Class 8: TP = 848, FP = 135, FN = 126, TN = 8891
Precision = 0.8626653
Recall(Sensitivity) = 0.8706365
F1-score: = 0.8666326
---------------------------------------------
Class 9: TP = 919, FP = 95, FN = 90, TN = 8896
Precision = 0.90631163
Recall(Sensitivity) = 0.9108028
F1-score: = 0.9085517
---------------------------------------------

Loss: 0.029155167
Accuracy: 91.88%
```


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