use std::collections::VecDeque;

use crate::serial::Layer;
use matrix_library::{Matrix, MatrixError};

use micrograd::{cell_ptr::CellPtr, node::Node};
use rand::{distributions::Distribution, thread_rng};
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
        Ok(self.w.matmul(x)? + self.b.clone())
    }

    fn params(&self) -> Vec<CellPtr> {
        let mut v: Vec<CellPtr> = Vec::new();
        // note: below calls to clone are Rc::clone() under the hood, so low memory cost
        v.append(
            self.w
                .clone()
                .into_iter()
                .map(|node| node.leaf().expect("all layer params are leaves").clone())
                .collect::<Vec<CellPtr>>()
                .as_mut(),
        );
        v.append(
            self.b
                .clone()
                .into_iter()
                .map(|node| node.leaf().expect("all layer params are leaves").clone())
                .collect::<Vec<CellPtr>>()
                .as_mut(),
        );
        v
    }
}

impl DenseLayer {
    pub fn new(output_size: usize, input_size: usize) -> DenseLayer {
        let mut rng = thread_rng();
        // He weight initialisation (performs well with ReLu)
        // https://arxiv.org/abs/1502.01852
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
        assert_eq!((3, 5), layer.w.shape());
        assert_eq!((3, 1), layer.b.shape());
    }

    #[test]
    fn forward() {
        let layer = DenseLayer::new(2, 3);
        let x_row = VecDeque::from([
            Node::from_f64(2.0),
            Node::from_f64(3.0),
            Node::from_f64(3.0),
        ]);
        let mut x_vec = VecDeque::new();
        x_vec.push_back(x_row);
        let x = Matrix::new(x_vec).transpose();

        let out = layer.forward(&x).unwrap();
        assert_eq!((2, 1), out.shape());
        let ans = layer.w.at((0, 0)).unwrap().resolve() * x.at((0, 0)).unwrap().resolve()
            + layer.w.at((0, 1)).unwrap().resolve() * x.at((1, 0)).unwrap().resolve()
            + layer.w.at((0, 2)).unwrap().resolve() * x.at((2, 0)).unwrap().resolve();
        assert_eq!(ans, out.at((0, 0)).unwrap().resolve());

        let ans_2 = layer.w.at((1, 0)).unwrap().resolve() * x.at((0, 0)).unwrap().resolve()
            + layer.w.at((1, 1)).unwrap().resolve() * x.at((1, 0)).unwrap().resolve()
            + layer.w.at((1, 2)).unwrap().resolve() * x.at((2, 0)).unwrap().resolve();
        assert_eq!(ans_2, out.at((1, 0)).unwrap().resolve());
    }
}
