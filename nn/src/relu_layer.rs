use std::collections::VecDeque;

use crate::serial::Layer;
use matrix_library::{Matrix, MatrixError};
use micrograd::node::Node;

pub struct ReluLayer {}

impl Layer for ReluLayer {
    fn forward(&self, x: &Matrix<Node>) -> Result<Matrix<Node>, MatrixError> {
        let mut res = VecDeque::new();
        for j in 0..x.shape().0 {
            let mut row = VecDeque::new();
            for i in 0..x.shape().1 {
                let el = x.at((j, i)).unwrap();
                if el.resolve() > 0.0 {
                    row.push_back(el.clone());
                } else {
                    row.push_back(Node::from_f64(0.0));
                }
            }
            res.push_back(row);
        }
        Ok(Matrix::new(res))
    }
}

impl ReluLayer {
    pub fn new() -> ReluLayer {
        ReluLayer {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let layer = ReluLayer::new();
        let x_row = VecDeque::from([
            Node::from_f64(-2.0),
            Node::from_f64(3.0),
            Node::from_f64(-3.0),
        ]);
        let mut x_vec = VecDeque::new();
        x_vec.push_back(x_row);
        let x = Matrix::new(x_vec).transpose();
        let out = layer.forward(&x).unwrap();

        //calc out manually
        assert_eq!(out.at((0, 0)).unwrap().resolve(), 0.0);
        assert_eq!(out.at((1, 0)).unwrap().resolve(), 3.0);
        assert_eq!(out.at((2, 0)).unwrap().resolve(), 0.0);
    }
}
