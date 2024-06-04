use std::collections::VecDeque;

// use crate::serial::Layer;
use interfaces::{ActivationLayer, DLModule, MathPrimitive, MathTensor, Primitive, Tensor};
use matrix_library::{Matrix, MatrixError};
use micrograd::{cell_ptr::CellPtr, node::Node};

pub struct ReluLayer {}

impl<T, P> ActivationLayer<T, P> for ReluLayer
where
    T: MathTensor<P>,
    P: MathPrimitive,
{
}

impl<T: MathTensor<P>, P: MathPrimitive> DLModule<T, P> for ReluLayer {
    type DLModuleError = <T as Tensor<P>>::TensorError;
    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        let mut res = Vec::new();
        for j in 0..x.shape()[0] {
            let mut row = Vec::new();
            for i in 0..x.shape()[1] {
                let el = x.at(vec![j, i]).unwrap();
                if el.as_f64() > 0.0 {
                    row.push(el.clone());
                } else {
                    row.push(P::from_f64(0.0));
                }
            }
            res.push(row);
        }
        Ok(T::from_vec(res))
    }

    fn params(&self) -> Vec<P> {
        vec![]
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
        assert_eq!(out.at(vec![0, 0]).unwrap().resolve(), 0.0);
        assert_eq!(out.at(vec![1, 0]).unwrap().resolve(), 3.0);
        assert_eq!(out.at(vec![2, 0]).unwrap().resolve(), 0.0);
    }
}
