use matrix_library::{Matrix, MatrixError};
use micrograd::node::Node;

use crate::dense_layer::DenseLayer;

pub trait Layer {
    fn forward(&self, x: &Matrix<Node>) -> Result<Matrix<Node>, MatrixError>;
}

pub struct Serial {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Serial {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Serial {
        Serial { layers }
    }
    pub fn forward(&self, x: &Matrix<Node>) -> Result<Matrix<Node>, MatrixError> {
        let mut tmp = x;
        // dummy
        let mut y = Matrix::fill((1, 1), Node::from_f64(0.0));
        for l in self.layers.iter() {
            y = l.forward(tmp)?;
            tmp = &y;
        }
        Ok(y)
    }
}
