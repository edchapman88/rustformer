use std::cell::RefCell;
use std::rc::Rc;
use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};

#[derive(PartialEq, Debug, Clone)]
pub struct Cell {
    val: RefCell<Value>,
}

impl Cell {
    pub fn new(data: f64) -> Cell {
        Cell {
            val: RefCell::new(Value::new(data)),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct CellPtr {
    ptr: Rc<Cell>,
}
impl CellPtr {
    pub fn new(cell: Rc<Cell>) -> CellPtr {
        CellPtr { ptr: cell }
    }
    pub fn from_f64(data: f64) -> CellPtr {
        CellPtr {
            ptr: Rc::new(Cell::new(data)),
        }
    }
    pub fn resolve(&self) -> f64 {
        self.ptr.val.borrow().resolve()
    }
    pub fn data_ref(&self) -> f64 {
        self.ptr.val.borrow().resolve()
    }
    pub fn grad_ref(&self) -> f64 {
        self.ptr.val.borrow().grad
    }
    pub fn add_grad(&self, g: f64) {
        self.ptr.val.borrow_mut().grad += g
    }
    pub fn clone(cell_ptr: &CellPtr) -> CellPtr {
        CellPtr {
            ptr: Rc::clone(&cell_ptr.ptr),
        }
    }
}
#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
}

impl Value {
    pub fn new(data: f64) -> Value {
        Value { data, grad: 0.0 }
    }
    fn resolve(&self) -> f64 {
        self.data
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Node {
    op: NodeOp,
    label: String,
}

#[derive(PartialEq, Debug, Clone)]
pub enum NodeOp {
    Add(Box<(NodeChild, NodeChild)>),
    Sub(Box<(NodeChild, NodeChild)>),
    Mul(Box<(NodeChild, NodeChild)>),
    Pow(Box<(NodeChild, NodeChild)>),
    Ln(Box<NodeChild>),
}

#[derive(PartialEq, Debug, Clone)]
pub enum NodeChild {
    Node(Node),
    Leaf(CellPtr),
}

impl NodeChild {
    fn resolve(&self) -> f64 {
        match self {
            NodeChild::Leaf(cellptr) => cellptr.resolve(),
            NodeChild::Node(node) => node.resolve(),
        }
    }
}

impl Node {
    pub fn new(op: NodeOp, label: String) -> Node {
        Node { op, label }
    }

    pub fn placeHolder() -> Node {
        Node {
            op: NodeOp::Ln(Box::new(NodeChild::Leaf(CellPtr::new(Rc::new(Cell::new(
                0.0,
            )))))),
            label: String::from("dummy"),
        }
    }

    pub fn stringify(&self) -> String {
        // let mut res = String::from("\n     ");
        // res += &self.label;
        // res += "\n   /    \\ \n";
        let mut res = String::from("\n");

        match &self.op {
            NodeOp::Add(bxd_children)
            | NodeOp::Mul(bxd_children)
            | NodeOp::Pow(bxd_children)
            | NodeOp::Sub(bxd_children) => {
                let (l_ch, r_ch) = &*(*bxd_children);

                match l_ch {
                    NodeChild::Leaf(l_val) => {
                        match r_ch {
                            NodeChild::Leaf(r_val) => {
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
                            }
                            NodeChild::Node(r_node) => {
                                // l leaf, r node
                                res += "     ";
                                res += &self.label;
                                res += "\n   /   \\ \n";
                                res += &l_val.data_ref().to_string();
                                res += "[";
                                res += &l_val.grad_ref().to_string();
                                res += "]\n";
                                res += &r_node.stringify();
                            }
                        }
                    }
                    NodeChild::Node(l_node) => {
                        match r_ch {
                            NodeChild::Leaf(r_val) => {
                                // l node, r leaf
                                res += "     ";
                                res += &self.label;
                                res += "\n   /   \\ \n        ";
                                res += &r_val.data_ref().to_string();
                                res += "[";
                                res += &r_val.grad_ref().to_string();
                                res += "]\n";
                                res += &l_node.stringify();
                            }
                            NodeChild::Node(r_node) => {
                                // both node
                                res += "     ";
                                res += &self.label;
                                res += "\n   /   \\ \n";
                                res += &l_node.stringify();
                                res += "\n";
                                res += &r_node.stringify();
                            }
                        }
                    }
                }
            }
            NodeOp::Ln(bxd_child) => {
                let child = &*(*bxd_child);
                match child {
                    NodeChild::Leaf(_) => panic!(".ln() not implemented for Value struct"),
                    NodeChild::Node(node) => {
                        res += "     ";
                        res += &self.label;
                        res += "\n   /\n        ";
                        res += &node.stringify();
                    }
                }
            }
        }
        res
    }

    pub fn resolve(&self) -> f64 {
        // extract l,r child values regardless of operation
        let (l, r) = match &self.op {
            NodeOp::Add(bxd_children)
            | NodeOp::Mul(bxd_children)
            | NodeOp::Pow(bxd_children)
            | NodeOp::Sub(bxd_children) => {
                let (l_ch, r_ch) = &*(*bxd_children);
                // handle left child
                let l_val = match l_ch {
                    NodeChild::Leaf(cell) => cell.data_ref(),
                    NodeChild::Node(node) => node.resolve(),
                };

                // handle right child
                let r_val = match r_ch {
                    NodeChild::Leaf(cell) => cell.data_ref(),
                    NodeChild::Node(node) => node.resolve(),
                };

                (l_val, r_val)
            }
            NodeOp::Ln(bxd_child) => {
                let child = &*(*bxd_child);
                let resolved_child = match child {
                    NodeChild::Leaf(_) => panic!(".ln() not implemented for Value struct"),
                    NodeChild::Node(node) => node.resolve(),
                };
                return resolved_child.ln();
            }
        };

        // operation specific resolution on l,r
        match &self.op {
            NodeOp::Add(_) => l + r,
            NodeOp::Sub(_) => l - r,
            NodeOp::Mul(_) => l * r,
            NodeOp::Pow(_) => l.powf(r),
            NodeOp::Ln(_) => panic!("handled with early return from previous match"),
        }
    }
    // fn _get_leaves(&self, mut leaf_vec: Vec<Value>) -> Vec<Value> {
    //     // order of maths operation
    //     match &self.op {
    //         NodeOp::Add(bxd_children)
    //         | NodeOp::Sub(bxd_children)
    //         | NodeOp::Mul(bxd_children)
    //         | NodeOp::Pow(bxd_children) => {
    //             let (l_ch, r_ch) = &*(*bxd_children);
    //             // handle left child
    //             match l_ch {
    //                 NodeChild::Leaf(value) => {
    //                     leaf_vec.push(value.clone());
    //                 }
    //                 NodeChild::Node(node) => {
    //                     leaf_vec = node._get_leaves(leaf_vec);
    //                 }
    //             };

    //             // handle right child
    //             match r_ch {
    //                 NodeChild::Leaf(value) => {
    //                     leaf_vec.push(value.clone());
    //                 }
    //                 NodeChild::Node(node) => {
    //                     leaf_vec = node._get_leaves(leaf_vec);
    //                 }
    //             };
    //         }
    //         NodeOp::Ln(bxd_child) => {
    //             let child = &*(*bxd_child);
    //             match child {
    //                 NodeChild::Leaf(_) => panic!(".ln() not implemented for Value struct"),
    //                 NodeChild::Node(node) => {
    //                     leaf_vec = node._get_leaves(leaf_vec);
    //                 }
    //             }
    //         }
    //     };
    //     leaf_vec
    // }
    fn backward(&mut self, out_grad: f64) {
        match &mut self.op {
            NodeOp::Add(bxd_children) => {
                let (l_ch, r_ch) = &mut *(*bxd_children);
                match l_ch {
                    NodeChild::Leaf(cell) => cell.add_grad(out_grad),
                    NodeChild::Node(node) => node.backward(out_grad),
                };
                match r_ch {
                    NodeChild::Leaf(cell) => cell.add_grad(out_grad),
                    NodeChild::Node(node) => node.backward(out_grad),
                }
            }
            NodeOp::Sub(bxd_children) => {
                let (l_ch, r_ch) = &mut *(*bxd_children);
                match l_ch {
                    NodeChild::Leaf(cell) => cell.add_grad(out_grad),
                    NodeChild::Node(node) => node.backward(out_grad),
                };
                match r_ch {
                    NodeChild::Leaf(cell) => cell.add_grad(-out_grad),
                    NodeChild::Node(node) => node.backward(-out_grad),
                }
            }
            NodeOp::Mul(bxd_children) => {
                let (l_ch, r_ch) = &mut *(*bxd_children);
                match l_ch {
                    NodeChild::Leaf(cell) => cell.add_grad(r_ch.resolve() * out_grad),
                    NodeChild::Node(node) => node.backward(r_ch.resolve() * out_grad),
                };
                match r_ch {
                    NodeChild::Leaf(cell) => cell.add_grad(l_ch.resolve() * out_grad),
                    NodeChild::Node(node) => node.backward(l_ch.resolve() * out_grad),
                };
            }
            NodeOp::Pow(bxd_children) => {
                // implicit that r_ch is the exponent
                let (l_ch, r_ch) = &mut *(*bxd_children);
                let l_val = l_ch.resolve();
                let r_val = r_ch.resolve();
                match l_ch {
                    NodeChild::Leaf(cell) => {
                        cell.add_grad(r_val * l_val.powf(r_val - 1.0) * out_grad);
                    }
                    NodeChild::Node(node) => {
                        node.backward(r_val * l_val.powf(r_val - 1.0) * out_grad);
                    }
                };
                match r_ch {
                    NodeChild::Leaf(cell) => {
                        cell.add_grad(l_val.powf(r_val) * l_val.ln() * out_grad);
                    }
                    NodeChild::Node(node) => {
                        node.backward(l_val.powf(r_val) * l_val.ln() * out_grad)
                    }
                };
            }
            NodeOp::Ln(bxd_child) => {
                let child = &mut *(*bxd_child);
                let child_val = child.resolve();
                match child {
                    NodeChild::Leaf(_) => panic!(".ln() not implemented for Value struct"),
                    NodeChild::Node(node) => node.backward(out_grad / child_val),
                }
            }
        };
    }
}

// assume you can only do operations between Value objects or Nodes (not plain floats)
// every Value used in an operation becomes a leaf and will have a grad returned following
// a call to backward.
impl Add for CellPtr {
    type Output = Node;
    fn add(self, rhs: Self) -> Self::Output {
        Node::new(
            NodeOp::Add(Box::new((NodeChild::Leaf(self), NodeChild::Leaf(rhs)))),
            String::from("+"),
        )
    }
}

// add node to leaf
impl Add<Node> for CellPtr {
    type Output = Node;
    fn add(self, rhs: Node) -> Self::Output {
        Node::new(
            NodeOp::Add(Box::new((NodeChild::Leaf(self), NodeChild::Node(rhs)))),
            String::from("+"),
        )
    }
}

// add leaf to node
impl Add<CellPtr> for Node {
    type Output = Node;
    fn add(self, rhs: CellPtr) -> Self::Output {
        Node::new(
            NodeOp::Add(Box::new((NodeChild::Node(self), NodeChild::Leaf(rhs)))),
            String::from("+"),
        )
    }
}

impl Add for Node {
    type Output = Node;
    fn add(self, rhs: Node) -> Self::Output {
        Node::new(
            NodeOp::Add(Box::new((NodeChild::Node(self), NodeChild::Node(rhs)))),
            String::from("+"),
        )
    }
}

impl Sub for CellPtr {
    type Output = Node;
    fn sub(self, rhs: Self) -> Self::Output {
        Node::new(
            NodeOp::Sub(Box::new((NodeChild::Leaf(self), NodeChild::Leaf(rhs)))),
            String::from("-"),
        )
    }
}

// subtract node from leaf
impl Sub<Node> for CellPtr {
    type Output = Node;
    fn sub(self, rhs: Node) -> Self::Output {
        Node::new(
            NodeOp::Sub(Box::new((NodeChild::Leaf(self), NodeChild::Node(rhs)))),
            String::from("-"),
        )
    }
}

// subtract leaf from node
impl Sub<CellPtr> for Node {
    type Output = Node;
    fn sub(self, rhs: CellPtr) -> Self::Output {
        Node::new(
            NodeOp::Sub(Box::new((NodeChild::Node(self), NodeChild::Leaf(rhs)))),
            String::from("-"),
        )
    }
}

impl Sub for Node {
    type Output = Node;
    fn sub(self, rhs: Node) -> Self::Output {
        Node::new(
            NodeOp::Sub(Box::new((NodeChild::Node(self), NodeChild::Node(rhs)))),
            String::from("-"),
        )
    }
}

impl Mul for CellPtr {
    type Output = Node;
    fn mul(self, rhs: Self) -> Self::Output {
        Node::new(
            NodeOp::Mul(Box::new((NodeChild::Leaf(self), NodeChild::Leaf(rhs)))),
            String::from("*"),
        )
    }
}
// mul ptr by node
impl Mul<Node> for CellPtr {
    type Output = Node;
    fn mul(self, rhs: Node) -> Self::Output {
        Node::new(
            NodeOp::Mul(Box::new((NodeChild::Leaf(self), NodeChild::Node(rhs)))),
            String::from("*"),
        )
    }
}

// mul node by ptr
impl Mul<CellPtr> for Node {
    type Output = Node;
    fn mul(self, rhs: CellPtr) -> Self::Output {
        Node::new(
            NodeOp::Mul(Box::new((NodeChild::Node(self), NodeChild::Leaf(rhs)))),
            String::from("*"),
        )
    }
}

impl Mul for Node {
    type Output = Node;
    fn mul(self, rhs: Self) -> Self::Output {
        Node::new(
            NodeOp::Mul(Box::new((NodeChild::Node(self), NodeChild::Node(rhs)))),
            String::from("*"),
        )
    }
}

impl CellPtr {
    pub fn pow_val(self, e: CellPtr) -> Node {
        Node::new(
            NodeOp::Pow(Box::new((NodeChild::Leaf(self), NodeChild::Leaf(e)))),
            String::from("^"),
        )
    }
    pub fn pow_node(self, e: Node) -> Node {
        Node::new(
            NodeOp::Pow(Box::new((NodeChild::Leaf(self), NodeChild::Node(e)))),
            String::from("^"),
        )
    }
}

impl Node {
    pub fn pow_val(self, e: CellPtr) -> Node {
        Node::new(
            NodeOp::Pow(Box::new((NodeChild::Node(self), NodeChild::Leaf(e)))),
            String::from("^"),
        )
    }
    pub fn pow_node(self, e: Node) -> Node {
        Node::new(
            NodeOp::Pow(Box::new((NodeChild::Node(self), NodeChild::Node(e)))),
            String::from("^"),
        )
    }
    pub fn ln(self) -> Node {
        Node::new(
            NodeOp::Ln(Box::new(NodeChild::Node(self))),
            String::from("ln"),
        )
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
        let graph = CellPtr::from_f64(4.0)
            + (CellPtr::from_f64(2.0) * CellPtr::from_f64(3.0) + CellPtr::from_f64(4.0))
                * CellPtr::from_f64(5.0)
            + CellPtr::from_f64(7.0);
        println!(
            "\n-------------------------------------\n Printing graph for 4 + (2 * 3 + 4) * 5 + 7"
        );
        println!("{graph}\n-------------------------------------\n");
        assert_eq!(graph.resolve(), 61.0);
    }

    #[test]
    fn backward() {
        let x0 = CellPtr::from_f64(4.0);
        let x1 = CellPtr::from_f64(2.0);
        let x2 = CellPtr::from_f64(3.0);
        let x3 = CellPtr::from_f64(4.0);
        let x4 = CellPtr::from_f64(5.0);
        let x5 = CellPtr::from_f64(7.0);

        let mut graph = CellPtr::clone(&x0)
            + (CellPtr::clone(&x1) * CellPtr::clone(&x2) + CellPtr::clone(&x3))
                * CellPtr::clone(&x4)
            + CellPtr::clone(&x5);
        graph.backward(1.0);
        println!("\n-------------------------------------\n visualise after backward()");
        println!("{graph}\n");

        assert_eq!(x0.grad_ref(), 1.0);
        assert_eq!(x1.grad_ref(), 15.0);
        assert_eq!(x2.grad_ref(), 10.0);
        assert_eq!(x3.grad_ref(), 5.0);
        assert_eq!(x4.grad_ref(), 10.0);
        assert_eq!(x5.grad_ref(), 1.0);
    }

    #[test]
    fn exponentiate_node() {
        let x0 = CellPtr::from_f64(2.0);
        let x1 = CellPtr::from_f64(1.0);

        let mut graph = (CellPtr::clone(&x0) + CellPtr::clone(&x1))
            .pow_node(CellPtr::from_f64(2.0) + CellPtr::from_f64(1.0));
        graph.backward(1.0);
        println!("{graph}");

        assert_eq!(x0.grad_ref(), 27.0);
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
        let x0 = CellPtr::from_f64(2.0);
        let w0 = CellPtr::from_f64(8.0);
        let b0 = CellPtr::from_f64(4.0);
        let y0 = CellPtr::from_f64(1.0);

        let y_pred0 = CellPtr::clone(&x0) * CellPtr::clone(&w0) + CellPtr::clone(&b0);

        let mut graph = (y_pred0 - CellPtr::clone(&y0)).pow_val(CellPtr::from_f64(2.0));
        graph.backward(1.0);
        println!("{graph}");

        // d/dw = 1/n sum( -2 * x_i(y_i - (w_i * x_i - b_i)) )
        // d/db = 1/n sum( -2 * (y_i - (w_i * x_i - b_i)) )
        // for n = 1:
        // d/dw0 = -2 * x0(y0 - (w0 * x0 + b0)) = 76
        // d/db0 = -2 * (y0 - (w0 * x0 + b0)) = 38

        assert_eq!(w0.grad_ref(), 76.0);
        assert_eq!(b0.grad_ref(), 38.0);
    }

    #[test]
    fn bin_x_entropy_calc() {
        // -1 * [ y * (y_pred + 0.0001).ln()    +    (1 - y) * (1 - (y_pred - 0.0001)).ln() ]
        // where:
        // y = 1.0
        // y_pred = 0.9
        let y_pred = CellPtr::from_f64(0.9);
        let y = CellPtr::from_f64(1.0);

        let mut e = CellPtr::from_f64(-1.0)
            * (CellPtr::clone(&y) * (CellPtr::clone(&y_pred) + CellPtr::from_f64(0.0001)).ln()
                + (CellPtr::from_f64(1.0) - CellPtr::clone(&y))
                    * (CellPtr::from_f64(1.0)
                        - (CellPtr::clone(&y_pred) - CellPtr::from_f64(0.0001)))
                    .ln());
        println!("{e}");
        e.backward(1.0);

        assert_eq!(y_pred.grad_ref(), bce_prime(1.0, 0.9));

        fn bce_prime(y: f64, y_pred: f64) -> f64 {
            -((y / (y_pred + 0.0001)) + (y - 1.0) / (1.0 - (y_pred - 0.0001)))
        }
    }
}
