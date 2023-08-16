use matrix_library::{Matrix, MatrixError};
use micrograd::{cell_ptr::CellPtr, node::Node};

pub trait Layer {
    fn forward(&self, x: &Matrix<Node>) -> Result<Matrix<Node>, MatrixError>;
    fn params(&self) -> Vec<CellPtr>;
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

    pub fn params(&self) -> Vec<CellPtr> {
        self.layers
            .iter()
            .map(|l| l.as_ref().params())
            .reduce(|mut acc, mut params| {
                acc.append(&mut params);
                acc
            })
            .expect("expect at least one parameter in the model")
    }
}
