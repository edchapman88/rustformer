use std::collections::VecDeque;

use crate::serial::{Layer, LayerError};
use matrix_library::{Matrix, MatrixError};

use micrograd::{cell_ptr::CellPtr, node::Node};
use rand::{distributions::Distribution, thread_rng, Rng};
use statrs::distribution::Normal;

pub struct DenseLayer {
    pub i_size: usize,
    pub o_size: usize,
    w: Matrix<Node>,
    b: Matrix<Node>,
}

impl Layer for DenseLayer {
    fn forward(&self, x: &Matrix<Node>) -> Result<Matrix<Node>, MatrixError> {
        if x.shape().1 != 1 {
            panic!(
                "batch forward not supported for shape {:?}, x must be of shape (j,1)",
                x.shape()
            );
        }
        self.w.matmul(x)
    }
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> DenseLayer {
        let mut rng = thread_rng();
        let norm = Normal::new(0.0, (2.0 / (input_size as f64)).sqrt()).unwrap();

        let mut w = VecDeque::new();
        let mut b = VecDeque::new();
        for _ in 0..output_size {
            let mut row = VecDeque::new();
            for _ in 0..input_size {
                row.push_back(Node::from_f64(norm.sample(&mut rng)))
            }
            w.push_back(row);
            b.push_back(Node::from_f64(0.0));
        }
        let w_mat = Matrix::new(w);
        let mut b_2d = VecDeque::new();
        b_2d.push_back(b);
        let b_mat = Matrix::new(b_2d).transpose();
        DenseLayer {
            i_size: input_size,
            o_size: output_size,
            w: w_mat,
            b: b_mat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let layer = DenseLayer::new(3, 5);
        assert_eq!((5, 3), layer.w.shape());
        assert_eq!((5, 1), layer.b.shape());
    }

    #[test]
    fn forward() {
        let layer = DenseLayer::new(2, 3);
        let x_row = VecDeque::from([Node::from_f64(2.0), Node::from_f64(3.0)]);
        let mut x_vec = VecDeque::new();
        x_vec.push_back(x_row);
        let x = Matrix::new(x_vec).transpose();

        let out = layer.forward(&x).unwrap();
        assert_eq!((3, 1), out.shape());
    }
}
