# Rustygrad

A really simple neural network implemented in (non idiomatic) rust for educational purposes. The entire point of this is to see how simple neural networks can be, so using optimized libraries feels more intuitive.

## Usage

The core of this library is `Value`. It wraps an f32 and builds a computational graph under the hood when mathematical operations are done with it.

```rust
use rustygrad::engine::Value;

fn my_nn() {
    // inputs
    let x1 = Value::from(2.0);
    let x2 = Value::from(0.0);

    // weights
    let w1 = Value::from(-3.0);
    let w2 = Value::from(1.0);

    // bias of the neuron
    let b = Value::from(6.8813635870195432);

    // compute x1*w1 + x2*w2 + b
    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();
    let n = x1w1.clone() + x2w2.clone() + b.clone();

    // use tanh as the activation function for the output of this graph.
    let output = n.tanh();

    output.backward();

    println!("x1 grad: {}", x1.grad());
    println!("x2 grad: {}", x2.grad());
    println!("w1 grad: {}", w1.grad());
    println!("w2 grad: {}", w2.grad());
    println!("b grad: {}", b.grad());
    println!("x1w1 grad: {}", x1w1.grad());
    println!("x2w2 grad: {}", x2w2.grad());
    println!("n grad: {}", n.grad());
    println!("output grad: {}", output.grad());

    assert_eq!(x1.grad(), -1.5000215);
    assert_eq!(x2.grad(), 0.50000715);
    assert_eq!(w1.grad(), 1.0000143);
    assert_eq!(w2.grad(), 0.0);
    assert_eq!(b.grad(), 0.50000715);
    assert_eq!(n.grad(), 0.50000715);
}
```
