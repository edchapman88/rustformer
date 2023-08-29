use crate::serial::Layer;
use matrix_library::math_utils::{Exp, Pow};
use matrix_library::{Matrix, MatrixError};
use micrograd::cell_ptr::CellPtr;
use micrograd::node::Node;

pub struct SigmoidLayer {}

impl Layer for SigmoidLayer {
    fn forward(&self, x: &Matrix<Node>) -> Result<Matrix<Node>, MatrixError> {
        Ok(
            ((x.clone() * Node::from_f64(-1.0)).exp() + Node::from_f64(1.0))
                .pow(Node::from_f64(-1.0)),
        )
    }
    fn params(&self) -> Vec<CellPtr> {
        vec![]
    }
}

impl SigmoidLayer {
    pub fn new() -> SigmoidLayer {
        SigmoidLayer {}
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    #[test]
    fn forward() {
        let layer = SigmoidLayer::new();
        let x_row = VecDeque::from([
            Node::from_f64(2.0),
            Node::from_f64(-3.0),
            Node::from_f64(3.0),
            Node::from_f64(-30.0),
            Node::from_f64(30.0),
        ]);
        let mut x_vec = VecDeque::new();
        x_vec.push_back(x_row);
        let x = Matrix::new(x_vec).transpose();
        let out = layer.forward(&x).unwrap();

        // 1 / (1 + e^(-x))
        let ans = (1.0_f64 + (-2.0_f64).exp()).powf(-1.0_f64);
        assert_eq!(out.at((0, 0)).unwrap().resolve(), ans);

        assert_eq!(
            (out.at((1, 0)).unwrap().resolve() * 100000.0).round(),
            ((1.0_f64 + (3.0_f64).exp()).powf(-1.0_f64) * 100000.0).round()
        );
        println!("{:?}", out.map(|x| x.resolve()).collect::<Vec<f64>>());
    }
}
