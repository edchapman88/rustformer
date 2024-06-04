// use crate::serial::Layer;
use interfaces::{DLModule, LinearLayer, MathPrimitive, MathTensor, Primitive, Tensor};
use matrix_library::{Matrix, MatrixError};
use micrograd::{cell_ptr::CellPtr, node::Node};
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use statrs::distribution::Normal;
use std::collections::VecDeque;
use std::marker::PhantomData;

pub struct DenseLayer<T, P>
where
    T: Tensor<P>,
    P: Primitive,
{
    pub i_size: usize,
    pub o_size: usize,
    // w: Matrix<Node>,
    // b: Matrix<Node>,
    w: T,
    b: T,
    phantom_tensor_primitive: PhantomData<P>,
}

// #[derive(Error)]
// pub enum DenseLayerError<T: Tensor<P>, P> {
//     PlaceHolder,
//     TensorError(#[from] <T as Tensor<P>>::TensorError),
// }

impl<T, P> LinearLayer<T, P> for DenseLayer<T, P>
where
    T: Tensor<P>,
    P: Primitive,
{
}

impl<T: Tensor<P>, P: Primitive> DLModule<T, P> for DenseLayer<T, P> {
    type DLModuleError = <T as Tensor<P>>::TensorError;
    /// x.shape().1 must equal layer.i_size, returns (b, o_size)
    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        let o = self.w.matmul(&x.clone().transpose())? + self.b.clone();
        Ok(o.transpose())
    }

    fn params(&self) -> Vec<P> {
        let mut v: Vec<P> = Vec::new();
        // note: below calls to clone are Rc::clone() under the hood, so low memory cost
        v.append(
            self.w
                .clone()
                .into_iter()
                // .map(|node| node.leaf().expect("all layer params are leaves").clone())
                .collect::<Vec<P>>()
                .as_mut(),
        );
        v.append(
            self.b
                .clone()
                .into_iter()
                // .map(|node| node.leaf().expect("all layer params are leaves").clone())
                .collect::<Vec<P>>()
                .as_mut(),
        );
        v
    }
}

impl DenseLayer<Matrix<Node>, Node> {
    pub fn new(
        output_size: usize,
        input_size: usize,
        seed: Option<u64>,
    ) -> DenseLayer<Matrix<Node>, Node> {
        let mut rng = if let Some(seed_n) = seed {
            ChaCha8Rng::seed_from_u64(seed_n)
        } else {
            ChaCha8Rng::from_entropy()
        };
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
            phantom_tensor_primitive: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let layer = DenseLayer::new(3, 5, None);
        assert_eq!(vec![3, 5], layer.w.shape());
        assert_eq!(vec![3, 1], layer.b.shape());
    }

    #[test]
    fn forward() {
        let layer = DenseLayer::new(2, 3, None);
        let x_row = VecDeque::from([
            Node::from_f64(2.0),
            Node::from_f64(3.0),
            Node::from_f64(3.0),
        ]);
        let mut x_vec = VecDeque::new();
        x_vec.push_back(x_row);
        let x = Matrix::new(x_vec);

        let out = layer.forward(&x).unwrap();
        assert_eq!(vec![1, 2], out.shape());
        let ans = layer.w.at(vec![0, 0]).unwrap().resolve() * x.at(vec![0, 0]).unwrap().resolve()
            + layer.w.at(vec![0, 1]).unwrap().resolve() * x.at(vec![0, 1]).unwrap().resolve()
            + layer.w.at(vec![0, 2]).unwrap().resolve() * x.at(vec![0, 2]).unwrap().resolve();
        assert_eq!(ans, out.at(vec![0, 0]).unwrap().resolve());

        let ans_2 = layer.w.at(vec![1, 0]).unwrap().resolve() * x.at(vec![0, 0]).unwrap().resolve()
            + layer.w.at(vec![1, 1]).unwrap().resolve() * x.at(vec![0, 1]).unwrap().resolve()
            + layer.w.at(vec![1, 2]).unwrap().resolve() * x.at(vec![0, 2]).unwrap().resolve();
        assert_eq!(ans_2, out.at(vec![0, 1]).unwrap().resolve());
    }

    #[test]
    fn forward_batch() {
        let layer = DenseLayer::new(2, 3, None);
        let x_row_0 = VecDeque::from([
            Node::from_f64(2.0),
            Node::from_f64(3.0),
            Node::from_f64(3.0),
        ]);
        let x_row_1 = VecDeque::from([
            Node::from_f64(4.0),
            Node::from_f64(5.0),
            Node::from_f64(6.0),
        ]);
        let mut x_vec = VecDeque::new();
        x_vec.push_back(x_row_0);
        x_vec.push_back(x_row_1);
        let x = Matrix::new(x_vec);
        println!("x shape: {:?}", x.shape());

        let out = layer.forward(&x).unwrap();
        assert_eq!(vec![2, 2], out.shape());

        let ans_0_0 = layer.w.at(vec![0, 0]).unwrap().resolve()
            * x.at(vec![0, 0]).unwrap().resolve()
            + layer.w.at(vec![0, 1]).unwrap().resolve() * x.at(vec![0, 1]).unwrap().resolve()
            + layer.w.at(vec![0, 2]).unwrap().resolve() * x.at(vec![0, 2]).unwrap().resolve();

        let ans_0_1 = layer.w.at(vec![1, 0]).unwrap().resolve()
            * x.at(vec![0, 0]).unwrap().resolve()
            + layer.w.at(vec![1, 1]).unwrap().resolve() * x.at(vec![0, 1]).unwrap().resolve()
            + layer.w.at(vec![1, 2]).unwrap().resolve() * x.at(vec![0, 2]).unwrap().resolve();

        let ans_1_0 = layer.w.at(vec![0, 0]).unwrap().resolve()
            * x.at(vec![1, 0]).unwrap().resolve()
            + layer.w.at(vec![0, 1]).unwrap().resolve() * x.at(vec![1, 1]).unwrap().resolve()
            + layer.w.at(vec![0, 2]).unwrap().resolve() * x.at(vec![1, 2]).unwrap().resolve();

        let ans_1_1 = layer.w.at(vec![1, 0]).unwrap().resolve()
            * x.at(vec![1, 0]).unwrap().resolve()
            + layer.w.at(vec![1, 1]).unwrap().resolve() * x.at(vec![1, 1]).unwrap().resolve()
            + layer.w.at(vec![1, 2]).unwrap().resolve() * x.at(vec![1, 2]).unwrap().resolve();

        assert_eq!(ans_0_0, out.at(vec![0, 0]).unwrap().resolve());
        assert_eq!(ans_0_1, out.at(vec![0, 1]).unwrap().resolve());
        assert_eq!(ans_1_0, out.at(vec![1, 0]).unwrap().resolve());
        assert_eq!(ans_1_1, out.at(vec![1, 1]).unwrap().resolve());
    }
}
