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
    Exp,
    Pow(f32),
}

#[derive(Debug)]
struct ValueData {
    data: f32,
    grad: f32,
    op: Option<ValueOp>,
    children: (Option<Value>, Option<Value>),
    visited: bool,
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
            visited: false,
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

    pub fn exp(&self) -> Value {
        Value {
            data: Rc::new(RefCell::new(ValueData::new(
                self.data().exp(),
                Some(ValueOp::Exp),
                (Some(self.clone()), None),
            ))),
        }
    }

    fn visited(&self) -> bool {
        self.data.deref().borrow().visited
    }

    fn set_visited(&self, visited: bool) {
        self.data.borrow_mut().visited = visited;
    }

    pub fn backward(&self) {
        // this function sets all the gradients to zero too, totally preparing for the
        // process of back propagation.
        self.topological_sort()
            .iter_mut()
            .rev()
            .filter(|n| n.op().is_some())
            .for_each(|node| {
                let (child_a, child_b) = node.children();

                match node.op().unwrap() {
                    ValueOp::Add => {
                        child_a.unwrap().accumulate_grad(node.grad());
                        child_b.unwrap().accumulate_grad(node.grad());
                    }
                    ValueOp::Mul => {
                        let a = child_a.unwrap();
                        let b = child_b.unwrap();
                        a.accumulate_grad(b.data() * node.grad());
                        b.accumulate_grad(a.data() * node.grad());
                    }
                    ValueOp::TanH => {
                        let a = child_a.unwrap();
                        let ddx = ddx_tanh(a.data());
                        a.accumulate_grad(ddx * node.grad());
                    }
                    ValueOp::Exp => {
                        let a = child_a.unwrap();
                        a.accumulate_grad(a.data().exp() * node.grad());
                    }
                    ValueOp::Pow(n) => {
                        let a = child_a.unwrap();
                        let ddx = n * a.data().powf(n - 1.0);
                        a.accumulate_grad(ddx * node.grad());
                    }
                }
            });
    }

    fn topological_sort(&self) -> Vec<Value> {
        let mut result = Vec::new();
        let mut stack = vec![Some(self.clone())];

        // reset all the nodes to not visited.
        while !stack.is_empty() {
            if let Some(node) = stack.pop().unwrap_or(None) {
                node.set_visited(false);
                node.set_grad(0.0);
                let (a, b) = node.children();
                stack.push(a);
                stack.push(b);
            }
        }
        self.set_grad(1.0);

        // helper function to determine if node is edge node
        let is_edge_node = |node: &Value| {
            let children = node.children();

            // if child 0 doesn't exist or is visited, set true
            let a_visited = if let Some(child) = children.0 {
                child.visited()
            } else {
                true
            };

            // if child b doesn't exist or is visited set true
            let b_visited = if let Some(child) = children.1 {
                child.visited()
            } else {
                true
            };

            // if both are true, this node has no children or they are all visited.
            a_visited && b_visited
        };

        // do the topological sort
        stack.push(Some(self.clone()));
        while !stack.is_empty() {
            if let Some(node) = stack.pop().unwrap() {
                let is_edge_node = is_edge_node(&node);
                let children = node.children();
                // if its not an edge node, push it back on the stack and process the children.
                // This does an iterative DFS on the graph (mimicking recursion).
                if !is_edge_node {
                    stack.push(Some(node));
                    if children.0.is_some() {
                        stack.push(children.0);
                    }
                    if children.1.is_some() {
                        stack.push(children.1);
                    }
                } else if !node.visited() {
                    node.set_visited(true);
                    result.push(node);
                }
            }
        }

        result
    }

    pub fn set_grad(&self, grad: f32) {
        self.data.borrow_mut().grad = grad;
    }

    pub fn accumulate_grad(&self, grad: f32) {
        self.data.borrow_mut().grad += grad;
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

impl ops::Add<f32> for Value {
    type Output = Value;

    fn add(self, rhs: f32) -> Self::Output {
        self + Value::from(rhs)
    }
}

impl ops::Add<Value> for f32 {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        rhs + Value::from(self)
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

impl ops::Mul<f32> for Value {
    type Output = Value;

    fn mul(self, rhs: f32) -> Self::Output {
        self * Value::from(rhs)
    }
}

impl ops::Mul<Value> for f32 {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        rhs * Value::from(self)
    }
}

impl ops::Sub<Value> for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        self + (-rhs)
    }
}

impl ops::Sub<f32> for Value {
    type Output = Value;

    fn sub(self, rhs: f32) -> Self::Output {
        self - Value::from(rhs)
    }
}

impl ops::Sub<Value> for f32 {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        Value::from(self) - rhs
    }
}

impl ops::Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl ops::BitXor<f32> for Value {
    type Output = Value;

    fn bitxor(self, rhs: f32) -> Self::Output {
        Value {
            data: Rc::new(RefCell::new(ValueData::new(
                self.data().powf(rhs),
                Some(ValueOp::Pow(rhs)),
                (Some(self), None),
            ))),
        }
    }
}

/// Defining division this way because it makes the derivative easy.
impl ops::Div<Value> for Value {
    type Output = Value;

    fn div(self, rhs: Value) -> Self::Output {
        self * (rhs ^ -1.0)
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
    _ = Value::from(42.0);
}

#[test]
fn value_add() {
    let a = Value::from(10.0);
    let b = Value::from(11.0);
    let c = a + b;
    assert_eq!(c.data(), 21.0);
}

#[test]
fn value_mul() {
    let a = Value::from(2.0);
    let b = Value::from(3.0);
    let c = a * b;
    assert_eq!(c.data(), 6.0);
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

    assert_eq!(x1.grad(), -1.5000215);
    assert_eq!(x2.grad(), 0.50000715);
    assert_eq!(w1.grad(), 1.0000143);
    assert_eq!(w2.grad(), 0.0);
    assert_eq!(b.grad(), 0.50000715);
    assert_eq!(n.grad(), 0.50000715);
}

#[test]
fn value_backward_tanh_parts() {
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
    // except in this test its hand made to test other operations.
    let e = (2.0 * n.clone()).exp();
    let output = (e.clone() - 1.0) / (e.clone() + 1.0);

    output.backward();

    assert_eq!(x1.grad(), -1.5000217);
    assert_eq!(x2.grad(), 0.5000072);
    assert_eq!(w1.grad(), 1.0000144);
    assert_eq!(w2.grad(), 0.0);
    assert_eq!(b.grad(), 0.5000072);
    assert_eq!(n.grad(), 0.5000072);
}

#[test]
fn value_backward_tanh_div() {
    let n = Value::from(0.8813634);
    let e = (2.0 * n.clone()).exp();
    let output = (e.clone() - 1.0) / (e.clone() + 1.0);

    output.backward();

    assert_eq!(n.grad(), 0.5000072);
    assert_eq!(e.grad(), 0.042894714);
    assert_eq!(output.data(), 0.7071017);
}

#[test]
fn value_backward_multi_connected() {
    let a = Value::from(3.0);
    let b = a.clone() + a.clone();

    b.backward();

    assert_eq!(a.grad(), 2.0);
}

#[test]
fn value_backward_multi_connected_complex() {
    let a = Value::from(-2.0);
    let b = Value::from(3.0);
    let d = a.clone() * b.clone();
    let e = a.clone() + b.clone();
    let f = d.clone() * e.clone();

    f.backward();

    assert_eq!(a.grad(), -3.0);
    assert_eq!(b.grad(), -8.0);
}

#[test]
fn value_backward_exp() {
    let a = Value::from(2.0);
    let b = a.exp();

    b.backward();

    assert_eq!(a.grad(), b.data());
}

#[test]
fn value_backward_pow() {
    let a = Value::from(2.0);
    let b = a.clone() ^ 3.0;

    b.backward();

    assert_eq!(a.grad(), 12.0);
    assert_eq!(b.data(), 8.0);
}

#[test]
fn value_backward_div() {
    let a = Value::from(4.0);
    let b = Value::from(2.0);
    let c = a.clone() / b.clone();

    c.backward();

    assert_eq!(a.grad(), 0.5);
    assert_eq!(b.grad(), -1.0);
}
