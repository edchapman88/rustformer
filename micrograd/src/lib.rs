use std::{ops::{Add, Mul}, fmt::Display};

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
}

impl Value {
    pub fn new(data: f64) -> Value {
        Value { data, grad:0.0 }
    }
    fn resolve(&self) -> f64 {
        self.data
    }
}

#[derive(PartialEq, Debug)]
pub struct Node {
    op: NodeOp,
    label: String
}

#[derive(PartialEq, Debug)]
pub enum NodeOp {
    Add(Box<(NodeChild,NodeChild)>),
    Mul(Box<(NodeChild,NodeChild)>),
}

#[derive(PartialEq, Debug)]
pub enum NodeChild {
    Node(Node),
    Leaf(Value)
}

impl NodeChild {
    fn resolve(&self) -> f64 {
        match self {
            NodeChild::Leaf(value) => value.resolve(),
            NodeChild::Node(node) => node.resolve()
        }
    }
}


impl Node {
    pub fn new(op:NodeOp,label:String) -> Node {
        Node { op, label }
    }

    pub fn stringify(&self) -> String {
        // let mut res = String::from("\n     ");
        // res += &self.label;
        // res += "\n   /    \\ \n";
        let mut res = String::from("\n");

        match &self.op {
            NodeOp::Add(bxd_children) | NodeOp::Mul(bxd_children) => {
                let (l_ch,r_ch) = &*(*bxd_children);

                match l_ch {
                    NodeChild::Leaf(l_val) => {
                        match r_ch {
                            NodeChild::Leaf(r_val) => {
                                // both leaf
                                res += "     ";
                                res += &self.label;
                                res += "\n   /   \\ \n";
                                res += &l_val.data.to_string();
                                res += "[";
                                res += &l_val.grad.to_string();
                                res += "]";
                                res += "   ";
                                res += &r_val.data.to_string();
                                res += "[";
                                res += &r_val.grad.to_string();
                                res += "]";
                            },
                            NodeChild::Node(r_node) => {
                                // l leaf, r node
                                res += "     ";
                                res += &self.label;
                                res += "\n   /   \\ \n";
                                res += &l_val.data.to_string();
                                res += "[";
                                res += &l_val.grad.to_string();
                                res += "]\n";
                                res += &r_node.stringify();

                            }
                        }
                    },
                    NodeChild::Node(l_node) => {
                        match r_ch {
                            NodeChild::Leaf(r_val) => {
                                // l node, r leaf
                                res += "     ";
                                res += &self.label;
                                res += "\n   /   \\ \n        ";
                                res += &r_val.data.to_string();
                                res += "[";
                                res += &r_val.grad.to_string();
                                res += "]\n";
                                res += &l_node.stringify();
                            },
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
        }
        res
    }

    pub fn resolve(&self) -> f64 {
        // extract l,r child values regardless of operation
        let (l,r) = match &self.op {
            NodeOp::Add(bxd_children) | NodeOp::Mul(bxd_children) => {
                let (l_ch,r_ch) = &*(*bxd_children);
                // handle left child
                let l_val = match l_ch {
                    NodeChild::Leaf(value) => {
                        value.data
                    },
                    NodeChild::Node(node) => {
                        node.resolve()
                    }
                };

                // handle right child
                let r_val = match r_ch {
                    NodeChild::Leaf(value) => {
                        value.data
                    },
                    NodeChild::Node(node) => {
                        node.resolve()
                    }
                };

                (l_val,r_val)
            },
        };

        // operation specific resolution on l,r
        match &self.op {
            NodeOp::Add(_) => l + r,
            NodeOp::Mul(_) => l * r
        }
    }

    // ONLY ROBUST TO UNBRANCHED TREES
    pub fn backward(&mut self, out_grad: f64) -> Vec<Value> {
        // calls _backward recursively (working down the tree)
        // and then calls _get_leaves() to return vec of leaves in
        // order of math operations
        // d * ( a + b ) + c => [d',a',b',c']
        // where ' means grad
        self._backward(out_grad);
        let leaf_vec = Vec::new();
        self._get_leaves(leaf_vec)
    }
    fn _get_leaves(&self, mut leaf_vec:Vec<Value>) -> Vec<Value> {
        // order of maths operation
        match &self.op {
            NodeOp::Add(bxd_children) | NodeOp::Mul(bxd_children) => {
                let (l_ch,r_ch) = &*(*bxd_children);
                // handle left child
                match l_ch {
                    NodeChild::Leaf(value) => {
                        leaf_vec.push(value.clone());
                    },
                    NodeChild::Node(node) => {
                        leaf_vec = node._get_leaves(leaf_vec);
                    }
                };

                // handle right child
                match r_ch {
                    NodeChild::Leaf(value) => {
                        leaf_vec.push(value.clone());
                    },
                    NodeChild::Node(node) => {
                        leaf_vec = node._get_leaves(leaf_vec);
                    }
                };
            }
        };
        leaf_vec
    }
    fn _backward(&mut self, out_grad: f64) {
        match &mut self.op {
            NodeOp::Add(bxd_children) => {
                let (l_ch,r_ch) = &mut*(*bxd_children);
                match l_ch {
                    NodeChild::Leaf(value) => value.grad = out_grad,
                    NodeChild::Node(node) => node._backward(out_grad)
                };
                match r_ch {
                    NodeChild::Leaf(value) => value.grad = out_grad,
                    NodeChild::Node(node) => node._backward(out_grad)
                }
            },
            NodeOp::Mul(bxd_children) => {
                let (l_ch,r_ch) = &mut*(*bxd_children);
                match l_ch {
                    NodeChild::Leaf(value) => value.grad = r_ch.resolve() * out_grad,
                    NodeChild::Node(node) => node._backward(r_ch.resolve() * out_grad)
                };
                match r_ch {
                    NodeChild::Leaf(value) => value.grad = l_ch.resolve() * out_grad,
                    NodeChild::Node(node) => node._backward(l_ch.resolve() * out_grad)
                };
            }
        };
    }
}

// impl Node {
    // pub fn backward(&self, out_grad: f64) -> (Value,Value) {
    //     match self {
    //         Node::Add((l,r)) => {
    //             (Value {data:*l, grad:Some(out_grad)},
    //                 Value {data:*r, grad:Some(out_grad)})
    //         },
    //         Node::Mul((l,r)) => {
    //             (Value {data:*l, grad: Some(r * out_grad)},
    //                 Value {data:*r, grad: Some(l * out_grad)})
    //         }
    //     }
    // }
// }




// assume you can only do operations between Value objects or Nodes (not plain floats)
// every Value used in an operation becomes a leaf and will have a grad returned following
// a call to backward.
impl Add for &Value {
    type Output = Node;
    fn add(self, rhs: Self) -> Self::Output {
        Node::new(
            NodeOp::Add(
                Box::new(
                    (
                        NodeChild::Leaf(Value::new(self.data)),
                        NodeChild::Leaf(Value::new(rhs.data))
                    )
                )
            ),
            String::from("+")
        )
    }
}

// add node to leaf
impl Add<Node> for &Value {
    type Output = Node;
    fn add(self, rhs: Node) -> Self::Output {
        Node::new(
            NodeOp::Add(
                Box::new(
                    (
                        NodeChild::Leaf(Value::new(self.data)),
                        NodeChild::Node(rhs),
                    )
                )
            ),
            String::from("+")
        )
    }
}

// add leaf to node
impl Add<&Value> for Node {
    type Output = Node;
    fn add(self, rhs: &Value) -> Self::Output {
        Node::new(
            NodeOp::Add(
                Box::new(
                    (
                        NodeChild::Node(self),
                        NodeChild::Leaf(Value::new(rhs.data))
                    )
                )
            ),
            String::from("+")
        )
    }
}

impl Add for Node {
    type Output = Node;
    fn add(self, rhs: Node) -> Self::Output {
        Node::new(
            NodeOp::Add(
                Box::new(
                    (
                        NodeChild::Node(self),
                        NodeChild::Node(rhs)
                    )
                )
            ),
            String::from("+")
        )
    }
}

impl Mul for &Value {
    type Output = Node;
    fn mul(self, rhs: Self) -> Self::Output {
        Node::new(
            NodeOp::Mul(
                Box::new(
                    (
                        NodeChild::Leaf(Value::new(self.data)),
                        NodeChild::Leaf(Value::new(rhs.data))
                    )
                )),
             String::from("*")
        )
    }
}
// mul val by node
impl Mul<Node> for &Value {
    type Output = Node;
    fn mul(self, rhs: Node) -> Self::Output {
        Node::new(
            NodeOp::Mul(
                Box::new(
                    (
                        NodeChild::Leaf(Value::new(self.data)),
                        NodeChild::Node(rhs)
                        
                    )
                )
            ),
            String::from("*")
        )
    }
}

// mul node by val
impl Mul<&Value> for Node {
    type Output = Node;
    fn mul(self, rhs: &Value) -> Self::Output {
        Node::new(
            NodeOp::Mul(
                Box::new(
                    (
                        NodeChild::Node(self),
                        NodeChild::Leaf(Value::new(rhs.data))
                    )
                )
            ),
            String::from("*")
        )
    }
}

impl Mul for Node {
    type Output = Node;
    fn mul(self, rhs: Self) -> Self::Output {
        Node::new(
            NodeOp::Mul(
                Box::new(
                    (
                        NodeChild::Node(self),
                        NodeChild::Node(rhs)
                    )
                )
            ),
            String::from("*")
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
        let graph = &Value::new(4.0) + (&Value::new(2.0) * &Value::new(3.0) + &Value::new(4.0)) * &Value::new(5.0) + &Value::new(7.0);
        println!("\n-------------------------------------\n Printing graph for 4 + (2 + 3) * 5 + 7");
        println!("{graph}\n-------------------------------------\n");
        assert_eq!(graph.resolve(),61.0);
    }

    #[test]
    fn backward() {
        let mut graph = &Value::new(4.0) + (&Value::new(2.0) * &Value::new(3.0) + &Value::new(4.0))  * &Value::new(5.0) + &Value::new(7.0);
        let leaves = graph.backward(1.0);
        println!("\n-------------------------------------\n visualise after backward()");
        println!("{graph}\n-------------------------------------\n");
        assert_eq!(leaves,vec![
            Value { data: 4.0, grad: 1.0},
            Value { data: 2.0, grad: 15.0},
            Value { data: 3.0, grad: 10.0},
            Value { data: 4.0, grad: 5.0},
            Value { data: 5.0, grad: 10.0},
            Value { data: 7.0, grad: 1.0},])
    }
}
