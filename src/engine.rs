use std::{
    cell::RefCell,
    ops::{self, Deref},
    rc::Rc,
};

#[derive(Debug, Clone)]
enum ValueOp {
    Add,
    Mul,
    TanH,
}

#[derive(Debug)]
struct ValueData {
    data: f32,
    grad: f32,
    op: Option<ValueOp>,
    children: (Option<Value>, Option<Value>),
}

impl ValueData {
    fn from(data: f32) -> Self {
        Self::new(data, None, (None, None))
    }

    fn new(data: f32, op: Option<ValueOp>, children: (Option<Value>, Option<Value>)) -> Self {
        ValueData {
            data,
            grad: 0.0,
            op,
            children,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Value {
    data: Rc<RefCell<ValueData>>,
}

impl Value {
    pub fn from(data: f32) -> Self {
        Value {
            data: Rc::new(RefCell::new(ValueData::from(data))),
        }
    }

    pub fn data(&self) -> f32 {
        self.data.deref().borrow().data
    }

    pub fn tanh(&self) -> Value {
        Value {
            data: Rc::new(RefCell::new(ValueData::new(
                tanh(self.data()),
                Some(ValueOp::TanH),
                (Some(self.clone()), None),
            ))),
        }
    }

    pub fn backward(&self) {
        // sneaky sneaky borrow the graph as mut
        let mut value = self.data.borrow_mut();

        // calculate the derivative of this node with respect to itself:
        value.grad = 1.0;
        let mut nodes = vec![(
            value.op.clone(),
            value.grad,
            value.children.0.clone(),
            value.children.1.clone(),
        )];

        // loop over all of the children and calculate their gradients
        while !nodes.is_empty() {
            let (op, grad, a, b) = nodes.pop().unwrap();
            if op.is_none() {
                continue;
            }

            match op.unwrap() {
                ValueOp::Add => {
                    // copy a and b
                    let a = a.unwrap();
                    let b = b.unwrap();

                    // set grad for A and B
                    a.set_grad(grad);
                    b.set_grad(grad);

                    // push A and B onto the stack for processing.
                    let (ac1, ac2) = a.children();
                    nodes.push((a.op(), a.grad(), ac1, ac2));
                    let (bc1, bc2) = b.children();
                    nodes.push((b.op(), b.grad(), bc1, bc2));
                }
                ValueOp::Mul => {
                    // copy A and B
                    let a = a.unwrap();
                    let b = b.unwrap();

                    // update the gradients for A and B
                    a.set_grad(b.data() * grad);
                    b.set_grad(a.data() * grad);

                    // push A and B onto the stack for processing.
                    let (ac1, ac2) = a.children();
                    nodes.push((a.op(), a.grad(), ac1, ac2));
                    let (bc1, bc2) = b.children();
                    nodes.push((b.op(), b.grad(), bc1, bc2));
                }
                ValueOp::TanH => {
                    let a = a.unwrap();
                    let ddx = ddx_tanh(a.data());
                    a.set_grad(grad * ddx);
                    let (ac1, ac2) = a.children();
                    nodes.push((a.op(), a.grad(), ac1, ac2));
                }
            }
        }
    }

    pub fn set_grad(&self, grad: f32) {
        self.data.borrow_mut().grad = grad;
    }

    pub fn grad(&self) -> f32 {
        self.data.deref().borrow().grad
    }

    pub fn children(&self) -> (Option<Value>, Option<Value>) {
        self.data.deref().borrow().children.clone()
    }

    fn op(&self) -> Option<ValueOp> {
        self.data.deref().borrow().op.clone()
    }
}

impl ops::Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        Value {
            data: Rc::new(RefCell::new(ValueData::new(
                self.data() + rhs.data(),
                Some(ValueOp::Add),
                (Some(self), Some(rhs)),
            ))),
        }
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        Value {
            data: Rc::new(RefCell::new(ValueData::new(
                self.data() * rhs.data(),
                Some(ValueOp::Mul),
                (Some(self), Some(rhs)),
            ))),
        }
    }
}

fn tanh(x: f32) -> f32 {
    let numerator = std::f32::consts::E.powf(2.0 * x) - 1.0;
    let denominator = std::f32::consts::E.powf(2.0 * x) + 1.0;
    numerator / denominator
}

/// The derivative of tanh with respect to x
fn ddx_tanh(x: f32) -> f32 {
    let t = tanh(x);
    1.0 - (t * t)
}

#[test]
fn value_init() {
    let value = Value::from(42.0);
    println!("{:?}", value);
}

#[test]
fn value_add() {
    let a = Value::from(10.0);
    let b = Value::from(11.0);
    let c = a + b;
    assert_eq!(c.data(), 21.0);
    println!("{:?}", c)
}

#[test]
fn value_mul() {
    let a = Value::from(2.0);
    let b = Value::from(3.0);
    let c = a * b;
    assert_eq!(c.data(), 6.0);
    println!("{:?}", c);
}

#[test]
fn value_backward() {
    // this example if from https://www.youtube.com/watch?v=VMj-3S1tku0&t=476s
    // taken from around 50:00 in.
    let a = Value::from(2.0);
    let b = Value::from(-3.0);
    let c = Value::from(10.0);
    let e = a.clone() * b.clone();
    let d = e.clone() + c.clone();
    let f = Value::from(-2.0);
    let loss = d.clone() * f.clone();

    loss.backward();

    println!("a grad: {}", a.grad());
    println!("b grad: {}", b.grad());
    println!("c grad: {}", c.grad());
    println!("d grad: {}", d.grad());
    println!("e grad: {}", e.grad());
    println!("f grad: {}", f.grad());
    println!("loss grad: {}", loss.grad());

    assert_eq!(a.grad(), 6.0);
    assert_eq!(b.grad(), -4.0);
    assert_eq!(c.grad(), -2.0);
    assert_eq!(d.grad(), -2.0);
    assert_eq!(e.grad(), -2.0);
    assert_eq!(f.grad(), 4.0);
    assert_eq!(loss.grad(), 1.0);
}

#[test]
fn value_backward_tanh() {
    // taken from above video at 1:18:05
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