use matrix_library::math_utils::{Exp, Pow};
use matrix_library::Matrix;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::ops::{AddAssign, Div};
use std::rc::Rc;
use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};

use crate::cell_ptr::CellPtr;

#[derive(PartialEq, Debug, Clone)]
pub struct Node {
    op: Rc<NodeOp>,
    label: String,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Cache {
    inner: RefCell<Option<f64>>,
}

impl Cache {
    pub fn empty() -> Self {
        Cache {
            inner: RefCell::new(None),
        }
    }

    pub fn clear(&self) {
        *self.inner.borrow_mut() = None
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum NodeOp {
    Leaf(CellPtr),
    Add((Node, Node, Cache)),
    Sub((Node, Node, Cache)),
    Mul((Node, Node, Cache)),
    Pow((Node, Node, Cache)),
    Ln((Node, Cache)),
}

impl Node {
    pub fn new(op: NodeOp, label: String) -> Node {
        Node {
            op: Rc::new(op),
            label,
        }
    }

    pub fn from_f64(data: f64) -> Node {
        Node {
            op: Rc::new(NodeOp::Leaf(CellPtr::new(data))),
            label: '#'.to_string(),
        }
    }

    /// Fill a matrix with new Nodes (references to new data).
    /// Behaves differently to `Matrix::fill()` which clones the provided element - in the case where
    /// the element is a `Node`, the clone call creates a matrix of references to the Node provided as
    /// an argument to the function.
    pub fn fill_matrix_f64(shape: (usize, usize), data: f64) -> Matrix<Node> {
        let mut matrix = Vec::new();
        for _ in 0..shape.0 {
            let mut row = Vec::new();
            for _ in 0..shape.1 {
                row.push(Node::from_f64(data));
            }
            matrix.push(row);
        }
        Matrix::from_vecs(matrix)
    }

    pub fn leaf(&self) -> Option<&CellPtr> {
        if let NodeOp::Leaf(cellptr) = self.op.as_ref() {
            Some(cellptr)
        } else {
            None
        }
    }

    pub fn stringify(&self) -> String {
        // let mut res = String::from("\n     ");
        // res += &self.label;
        // res += "\n   /    \\ \n";
        let mut res = String::from("\n");

        match &self.op.as_ref() {
            NodeOp::Leaf(cellptr) => res += &cellptr.data_ref().to_string(),
            NodeOp::Add(children)
            | NodeOp::Mul(children)
            | NodeOp::Pow(children)
            | NodeOp::Sub(children) => {
                let (l_ch, r_ch, _) = children;

                if let NodeOp::Leaf(l_val) = l_ch.op.as_ref() {
                    if let NodeOp::Leaf(r_val) = r_ch.op.as_ref() {
                        // both leaf
                        res += "     ";
                        res += &self.label;
                        res += "\n   /   \\ \n";
                        res += &l_val.data_ref().to_string();
                        res += "[";
                        res += &l_val.grad_ref().to_string();
                        res += "]";
                        res += "   ";
                        res += &r_val.data_ref().to_string();
                        res += "[";
                        res += &r_val.grad_ref().to_string();
                        res += "]";
                    } else {
                        // l leaf, r node
                        res += "     ";
                        res += &self.label;
                        res += "\n   /   \\ \n";
                        res += &l_val.data_ref().to_string();
                        res += "[";
                        res += &l_val.grad_ref().to_string();
                        res += "]\n";
                        res += &r_ch.stringify();
                    }
                } else {
                    if let NodeOp::Leaf(r_val) = r_ch.op.as_ref() {
                        // l node, r leaf
                        res += "     ";
                        res += &self.label;
                        res += "\n   /   \\ \n        ";
                        res += &r_val.data_ref().to_string();
                        res += "[";
                        res += &r_val.grad_ref().to_string();
                        res += "]\n";
                        res += &l_ch.stringify();
                    } else {
                        // both node
                        res += "     ";
                        res += &self.label;
                        res += "\n   /   \\ \n";
                        res += &l_ch.stringify();
                        res += "\n";
                        res += &r_ch.stringify();
                    }
                }
            }
            NodeOp::Ln((child, _)) => {
                if let NodeOp::Leaf(_) = child.op.as_ref() {
                    panic!(".ln() not implemented for Value struct")
                } else {
                    res += "     ";
                    res += &self.label;
                    res += "\n   /\n        ";
                    res += &child.stringify();
                }
            }
        }
        res
    }

    pub fn resolve(&self) -> f64 {
        // extract l,r child values regardless of operation
        let (l, r) = match self.op.as_ref() {
            NodeOp::Leaf(cellptr) => {
                return cellptr.data_ref();
            }
            NodeOp::Add((l_ch, r_ch, cache))
            | NodeOp::Mul((l_ch, r_ch, cache))
            | NodeOp::Pow((l_ch, r_ch, cache))
            | NodeOp::Sub((l_ch, r_ch, cache)) => {
                if let Some(resolved) = cache.inner.borrow().as_ref() {
                    return *resolved;
                }
                (l_ch.resolve(), r_ch.resolve())
            }
            NodeOp::Ln((child, cache)) => {
                if let Some(resolved) = cache.inner.borrow().as_ref() {
                    return *resolved;
                }
                let resolution = child.resolve().ln();
                *cache.inner.borrow_mut() = Some(resolution);
                return resolution;
            }
        };

        // operation specific resolution on l,r
        let resolution = match self.op.as_ref() {
            NodeOp::Add(_) => l + r,
            NodeOp::Sub(_) => l - r,
            NodeOp::Mul(_) => l * r,
            NodeOp::Pow(_) => l.powf(r),
            NodeOp::Leaf(_) => panic!("handled with early return from previous match"),
            NodeOp::Ln(_) => panic!("handled with early return from previous match"),
        };

        // update cache
        match self.op.as_ref() {
            NodeOp::Add((_, _, cache))
            | NodeOp::Mul((_, _, cache))
            | NodeOp::Pow((_, _, cache))
            | NodeOp::Sub((_, _, cache)) => {
                *cache.inner.borrow_mut() = Some(resolution);
            }
            NodeOp::Leaf(_) => panic!("handled with early return from previous match"),
            NodeOp::Ln(_) => panic!("handled with early return from previous match"),
        };

        resolution
    }
    pub fn backward(&self, out_grad: f64) {
        match self.op.as_ref() {
            NodeOp::Leaf(cellptr) => cellptr.add_grad(out_grad),
            NodeOp::Add((l_ch, r_ch, _)) => {
                l_ch.backward(out_grad);
                r_ch.backward(out_grad);
            }
            NodeOp::Sub((l_ch, r_ch, _)) => {
                l_ch.backward(out_grad);
                r_ch.backward(-out_grad);
            }
            NodeOp::Mul((l_ch, r_ch, _)) => {
                l_ch.backward(r_ch.resolve() * out_grad);
                r_ch.backward(l_ch.resolve() * out_grad);
            }
            NodeOp::Pow((l_ch, r_ch, _)) => {
                // implicit that r_ch is the exponent
                let l_val = l_ch.resolve();
                let r_val = r_ch.resolve();
                l_ch.backward(r_val * l_val.powf(r_val - 1.0) * out_grad);
                r_ch.backward(l_val.powf(r_val) * l_val.ln() * out_grad);
            }
            NodeOp::Ln((child, _)) => {
                let child_val = child.resolve();
                child.backward(out_grad / child_val);
            }
        };
    }

    pub fn zero_grad(&self) {
        match self.op.as_ref() {
            NodeOp::Leaf(cellptr) => cellptr.zero_grad(),
            NodeOp::Add((l_ch, r_ch, _))
            | NodeOp::Sub((l_ch, r_ch, _))
            | NodeOp::Mul((l_ch, r_ch, _))
            | NodeOp::Pow((l_ch, r_ch, _)) => {
                l_ch.zero_grad();
                r_ch.zero_grad();
            }
            NodeOp::Ln((child, _)) => {
                child.zero_grad();
            }
        }
    }

    pub fn clear_all_caches(&self) {
        match self.op.as_ref() {
            NodeOp::Add((l_ch, r_ch, cache))
            | NodeOp::Sub((l_ch, r_ch, cache))
            | NodeOp::Mul((l_ch, r_ch, cache))
            | NodeOp::Pow((l_ch, r_ch, cache)) => {
                cache.clear();
                l_ch.clear_all_caches();
                r_ch.clear_all_caches();
            }
            NodeOp::Ln((child, cache)) => {
                cache.clear();
                child.clear_all_caches();
            }
            NodeOp::Leaf(_) => {}
        }
    }
}

impl Add for Node {
    type Output = Node;
    fn add(self, rhs: Node) -> Self::Output {
        Node::new(NodeOp::Add((self, rhs, Cache::empty())), String::from("+"))
    }
}

impl AddAssign for Node {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl Sub for Node {
    type Output = Node;
    fn sub(self, rhs: Node) -> Self::Output {
        Node::new(NodeOp::Sub((self, rhs, Cache::empty())), String::from("-"))
    }
}

impl Mul for Node {
    type Output = Node;
    fn mul(self, rhs: Self) -> Self::Output {
        Node::new(NodeOp::Mul((self, rhs, Cache::empty())), String::from("*"))
    }
}

impl Node {
    pub fn pow(self, e: Node) -> Node {
        Node::new(NodeOp::Pow((self, e, Cache::empty())), String::from("^"))
    }
    pub fn ln(self) -> Node {
        Node::new(NodeOp::Ln((self, Cache::empty())), String::from("ln"))
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.resolve())
    }
}

impl Exp for Node {
    fn exp(self) -> Self {
        Node::from_f64(std::f64::consts::E).pow(self)
    }
}

impl Pow for Node {
    fn pow(self, exp: Self) -> Self {
        self.pow(exp)
    }
}

impl Div for Node {
    type Output = Node;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(Node::from_f64(-1.0))
    }
}

#[cfg(test)]
mod tests {
    use matrix_library::Matrix;

    use super::*;

    #[test]
    fn build_and_resolve_graph() {
        let graph = Node::from_f64(4.0)
            + (Node::from_f64(2.0) * Node::from_f64(3.0) + Node::from_f64(4.0))
                * Node::from_f64(5.0)
            + Node::from_f64(7.0);
        println!(
            "\n-------------------------------------\n Printing graph for 4 + (2 * 3 + 4) * 5 + 7"
        );
        println!(
            "{}\n-------------------------------------\n",
            graph.stringify()
        );
        assert_eq!(graph.resolve(), 61.0);
    }

    #[test]
    fn add_assign() {
        let mut graph = Node::from_f64(2.0) * Node::from_f64(3.0);
        graph += Node::from_f64(4.0);
        println!("{}", graph.stringify());
        assert_eq!(10.0, graph.resolve());
    }

    #[test]
    fn backward() {
        let x0 = Node::from_f64(4.0);
        let x1 = Node::from_f64(2.0);
        let x2 = Node::from_f64(3.0);
        let x3 = Node::from_f64(4.0);
        let x4 = Node::from_f64(5.0);
        let x5 = Node::from_f64(7.0);

        // Derived Clone impl for Node uses custom Clone impl for CellPtr which takes a reference
        // using Rc::clone
        let mut graph =
            x0.clone() + (x1.clone() * x2.clone() + x3.clone()) * x4.clone() + x5.clone();
        graph.backward(1.0);
        println!("\n-------------------------------------\n visualise after backward()");
        println!("{}\n", graph.stringify());

        assert_eq!(x0.leaf().unwrap().grad_ref(), 1.0);
        assert_eq!(x1.leaf().unwrap().grad_ref(), 15.0);
        assert_eq!(x2.leaf().unwrap().grad_ref(), 10.0);
        assert_eq!(x3.leaf().unwrap().grad_ref(), 5.0);
        assert_eq!(x4.leaf().unwrap().grad_ref(), 10.0);
        assert_eq!(x5.leaf().unwrap().grad_ref(), 1.0);
    }

    #[test]
    fn zero_grad() {
        let x = Node::from_f64(3.2);
        let mut graph = (x.clone() + Node::from_f64(2.2)).pow(Node::from_f64(3.0).ln());
        graph.backward(1.0);
        assert_ne!(0.0, x.leaf().unwrap().grad_ref());
        graph.zero_grad();
        assert_eq!(0.0, x.leaf().unwrap().grad_ref());
    }

    #[test]
    fn backward_two_graphs_seperately() {
        let x = Node::from_f64(3.0);
        let five = Node::from_f64(5.0);
        let two = Node::from_f64(2.0);

        let mut graph1 = x.clone() * (five.clone() * two.clone()) + two.clone();
        let mut graph2 = x.clone() * (two.clone() + two.clone()) * two.clone();

        graph1.backward(1.0);
        assert_eq!(x.leaf().unwrap().grad_ref(), 10.0);
        graph2.backward(1.0);
        assert_eq!(x.leaf().unwrap().grad_ref(), 10.0 + 8.0);
    }

    #[test]
    fn exponentiate_node() {
        let x0 = Node::from_f64(2.0);
        let x1 = Node::from_f64(1.0);

        let mut graph = (x0.clone() + x1.clone()).pow(Node::from_f64(2.0) + Node::from_f64(1.0));
        graph.backward(1.0);
        println!("{}", graph.stringify());

        assert_eq!(x0.leaf().unwrap().grad_ref(), 27.0);
    }

    #[test]
    fn exponent_node() {
        let x0 = Node::from_f64(4.0);

        let mut graph = (Node::from_f64(3.0)).pow(Node::from_f64(3.0) - x0.clone());
        graph.backward(1.0);
        println!("{}", graph.stringify());

        // d/dx(3^(3-x)) = -3^(3-x) * ln(3)
        assert_eq!(
            x0.leaf().unwrap().grad_ref(),
            -(3.0_f64.powf(3.0 - 4.0)) * 3.0_f64.ln()
        );
    }

    #[test]
    fn raise_e_to_node() {
        let x0 = Node::from_f64(4.0);

        let mut graph = (Node::from_f64(3.0) - x0.clone()).exp();
        graph.backward(1.0);
        println!("{}", graph.stringify());

        // d/dx(e^(3-x)) = -e^(3-x)
        assert_eq!(
            x0.leaf().unwrap().grad_ref(),
            -(std::f64::consts::E.powf(3.0 - 4.0))
        );
    }

    #[test]
    fn mse_calc_tree() {
        // mse = 1/n * sum( (y_pred - y)^2 )
        // for n = 1:
        // ((x0*w0 + b0) - y0)^2 / 1
        // where:
        // x0 = 2
        // w0 = 8
        // b0 = 4
        // y0 = 1
        let x0 = Node::from_f64(2.0);
        let w0 = Node::from_f64(8.0);
        let b0 = Node::from_f64(4.0);
        let y0 = Node::from_f64(1.0);

        let y_pred0 = x0.clone() * w0.clone() + b0.clone();

        let mut graph = (y_pred0 - y0.clone()).pow(Node::from_f64(2.0));
        graph.backward(1.0);
        println!("{}", graph.stringify());

        // d/dw = 1/n sum( -2 * x_i(y_i - (w_i * x_i - b_i)) )
        // d/db = 1/n sum( -2 * (y_i - (w_i * x_i - b_i)) )
        // for n = 1:
        // d/dw0 = -2 * x0(y0 - (w0 * x0 + b0)) = 76
        // d/db0 = -2 * (y0 - (w0 * x0 + b0)) = 38

        assert_eq!(w0.leaf().unwrap().grad_ref(), 76.0);
        assert_eq!(b0.leaf().unwrap().grad_ref(), 38.0);
    }

    #[test]
    fn bin_x_entropy_calc() {
        // -1 * [ y * (y_pred + 0.0001).ln()    +    (1 - y) * (1 - (y_pred - 0.0001)).ln() ]
        // where:
        // y = 1.0
        // y_pred = 0.9
        let y_pred = Node::from_f64(0.9);
        let y = Node::from_f64(1.0);

        let mut e = Node::from_f64(-1.0)
            * (y.clone() * (y_pred.clone() + Node::from_f64(0.0001)).ln()
                + (Node::from_f64(1.0) - Node::clone(&y))
                    * (Node::from_f64(1.0) - (Node::clone(&y_pred) - Node::from_f64(0.0001))).ln());
        e.backward(1.0);
        println!("{e}");
        assert_eq!(y_pred.leaf().unwrap().grad_ref(), bce_prime(1.0, 0.9));

        fn bce_prime(y: f64, y_pred: f64) -> f64 {
            -((y / (y_pred + 0.0001)) + (y - 1.0) / (1.0 - (y_pred - 0.0001)))
        }
    }

    #[test]
    fn matrix_clone() {
        // ::fill() calls clone on the passed value to fill the matrix
        let a = Matrix::fill((2, 2), Node::from_f64(0.0));
        // since clone() is overridden for Node, now both "a" and "b" are filled with references to
        // the same Node (the Node passed in to the original ::fill() method)
        let b = a.clone();
        b.at((0, 0)).unwrap().leaf().unwrap().add_data(1.0);
        println! {"{}",b};
        assert_eq!(1.0, b.at((1, 1)).unwrap().leaf().unwrap().resolve());
        assert_eq!(1.0, a.at((1, 1)).unwrap().leaf().unwrap().resolve());
    }
}
