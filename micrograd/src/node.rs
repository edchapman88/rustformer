use std::rc::Rc;
use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};

use crate::cell_ptr::{Cell, CellPtr};

#[derive(PartialEq, Debug, Clone)]
pub struct Node {
    op: NodeOp,
    label: String,
}

#[derive(PartialEq, Debug, Clone)]
pub enum NodeOp {
    Leaf(CellPtr),
    Add(Box<(Node, Node)>),
    Sub(Box<(Node, Node)>),
    Mul(Box<(Node, Node)>),
    Pow(Box<(Node, Node)>),
    Ln(Box<Node>),
}

impl Node {
    pub fn new(op: NodeOp, label: String) -> Node {
        Node { op, label }
    }

    pub fn from_f64(data: f64) -> Node {
        Node {
            op: NodeOp::Leaf(CellPtr::new(data)),
            label: '#'.to_string(),
        }
    }

    pub fn leaf(&self) -> Option<&CellPtr> {
        if let NodeOp::Leaf(cellptr) = &self.op {
            Some(cellptr)
        } else {
            None
        }
    }

    // pub fn placeHolder() -> Node {
    //     Node {
    //         op: NodeOp::Ln(Box::new(NodeChild::Leaf(CellPtr::new(Rc::new(Cell::new(
    //             0.0,
    //         )))))),
    //         label: String::from("dummy"),
    //     }
    // }

    pub fn stringify(&self) -> String {
        // let mut res = String::from("\n     ");
        // res += &self.label;
        // res += "\n   /    \\ \n";
        let mut res = String::from("\n");

        match &self.op {
            NodeOp::Leaf(cellptr) => res += &cellptr.data_ref().to_string(),
            NodeOp::Add(bxd_children)
            | NodeOp::Mul(bxd_children)
            | NodeOp::Pow(bxd_children)
            | NodeOp::Sub(bxd_children) => {
                let (l_ch, r_ch) = &*(*bxd_children);

                if let NodeOp::Leaf(l_val) = &l_ch.op {
                    if let NodeOp::Leaf(r_val) = &r_ch.op {
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
                    if let NodeOp::Leaf(r_val) = &r_ch.op {
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
            NodeOp::Ln(bxd_child) => {
                let child = &*(*bxd_child);
                if let NodeOp::Leaf(_) = &child.op {
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
        let (l, r) = match &self.op {
            NodeOp::Leaf(cellptr) => return cellptr.data_ref(),
            NodeOp::Add(bxd_children)
            | NodeOp::Mul(bxd_children)
            | NodeOp::Pow(bxd_children)
            | NodeOp::Sub(bxd_children) => {
                let (l_ch, r_ch) = &*(*bxd_children);
                (l_ch.resolve(), r_ch.resolve())
            }
            NodeOp::Ln(bxd_child) => {
                let child = &*(*bxd_child);
                return child.resolve().ln();
            }
        };

        // operation specific resolution on l,r
        match &self.op {
            NodeOp::Add(_) => l + r,
            NodeOp::Sub(_) => l - r,
            NodeOp::Mul(_) => l * r,
            NodeOp::Pow(_) => l.powf(r),
            NodeOp::Leaf(_) => panic!("handled with early return from previous match"),
            NodeOp::Ln(_) => panic!("handled with early return from previous match"),
        }
    }
    fn backward(&mut self, out_grad: f64) {
        match &mut self.op {
            NodeOp::Leaf(cellptr) => cellptr.add_grad(out_grad),
            NodeOp::Add(bxd_children) => {
                let (l_ch, r_ch) = &mut *(*bxd_children);
                l_ch.backward(out_grad);
                r_ch.backward(out_grad);
            }
            NodeOp::Sub(bxd_children) => {
                let (l_ch, r_ch) = &mut *(*bxd_children);
                l_ch.backward(-out_grad);
                r_ch.backward(-out_grad);
            }
            NodeOp::Mul(bxd_children) => {
                let (l_ch, r_ch) = &mut *(*bxd_children);
                l_ch.backward(r_ch.resolve() * out_grad);
                r_ch.backward(l_ch.resolve() * out_grad);
            }
            NodeOp::Pow(bxd_children) => {
                // implicit that r_ch is the exponent
                let (l_ch, r_ch) = &mut *(*bxd_children);
                let l_val = l_ch.resolve();
                let r_val = r_ch.resolve();
                l_ch.backward(r_val * l_val.powf(r_val - 1.0) * out_grad);
                r_ch.backward(l_val.powf(r_val) * l_val.ln() * out_grad);
            }
            NodeOp::Ln(bxd_child) => {
                let child = &mut *(*bxd_child);
                let child_val = child.resolve();
                child.backward(out_grad / child_val);
            }
        };
    }
}

impl Add for Node {
    type Output = Node;
    fn add(self, rhs: Node) -> Self::Output {
        Node::new(NodeOp::Add(Box::new((self, rhs))), String::from("+"))
    }
}

impl Sub for Node {
    type Output = Node;
    fn sub(self, rhs: Node) -> Self::Output {
        Node::new(NodeOp::Sub(Box::new((self, rhs))), String::from("-"))
    }
}

impl Mul for Node {
    type Output = Node;
    fn mul(self, rhs: Self) -> Self::Output {
        Node::new(NodeOp::Mul(Box::new((self, rhs))), String::from("*"))
    }
}

impl Node {
    pub fn pow(self, e: Node) -> Node {
        Node::new(NodeOp::Pow(Box::new((self, e))), String::from("^"))
    }
    pub fn ln(self) -> Node {
        Node::new(NodeOp::Ln(Box::new(self)), String::from("ln"))
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.stringify())
    }
}

#[cfg(test)]
mod tests {
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
        println!("{graph}\n-------------------------------------\n");
        assert_eq!(graph.resolve(), 61.0);
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
        println!("{graph}\n");

        assert_eq!(x0.leaf().unwrap().grad_ref(), 1.0);
        assert_eq!(x1.leaf().unwrap().grad_ref(), 15.0);
        assert_eq!(x2.leaf().unwrap().grad_ref(), 10.0);
        assert_eq!(x3.leaf().unwrap().grad_ref(), 5.0);
        assert_eq!(x4.leaf().unwrap().grad_ref(), 10.0);
        assert_eq!(x5.leaf().unwrap().grad_ref(), 1.0);
    }

    #[test]
    fn exponentiate_node() {
        let x0 = Node::from_f64(2.0);
        let x1 = Node::from_f64(1.0);

        let mut graph = (x0.clone() + x1.clone()).pow(Node::from_f64(2.0) + Node::from_f64(1.0));
        graph.backward(1.0);
        println!("{graph}");

        assert_eq!(x0.leaf().unwrap().grad_ref(), 27.0);
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
        println!("{graph}");

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
        println!("{e}");
        e.backward(1.0);

        assert_eq!(y_pred.leaf().unwrap().grad_ref(), bce_prime(1.0, 0.9));

        fn bce_prime(y: f64, y_pred: f64) -> f64 {
            -((y / (y_pred + 0.0001)) + (y - 1.0) / (1.0 - (y_pred - 0.0001)))
        }
    }
}
