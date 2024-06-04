use interfaces::DLModule;
use interfaces::EmbeddingLayer;
use interfaces::MathPrimitive;
use interfaces::Primitive;
use interfaces::Tensor;
use matrix_library::{Matrix, MatrixError};
use micrograd::cell_ptr::CellPtr;
use micrograd::node::Node;
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use statrs::distribution::Normal;
use std::fmt::Display;
use std::marker::PhantomData;
pub struct EmbeddingTable<T, P>
where
    T: Tensor<P>,
    P: Primitive,
{
    // table: Matrix<Node>,
    table: T,
    phantom_tensor_primitive: PhantomData<P>,
}

impl<T, P> EmbeddingLayer<T, P> for EmbeddingTable<T, P>
where
    T: Tensor<P>,
    P: Primitive,
{
}

impl<T, P> DLModule<T, P> for EmbeddingTable<T, P>
where
    T: Tensor<P>,
    P: Primitive,
{
    type DLModuleError = <T as Tensor<P>>::TensorError;

    /// Retrieve a set of embeddings
    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        // if x.shape()[1] != self.table.shape()[0] {
        //     return Err(MatrixError::DimMismatch(x.shape(), self.table.shape()));
        // }
        x.matmul(&self.table)
    }

    fn params(&self) -> Vec<P> {
        let mut v: Vec<P> = Vec::new();
        v.append(
            self.table
                .clone()
                .into_iter()
                // .map(|node| node.leaf().expect("all layer params are leaves").clone())
                .collect::<Vec<P>>()
                .as_mut(),
        );
        v
    }
}

impl EmbeddingTable<Matrix<Node>, Node> {
    pub fn new(
        vocab_size: usize,
        emb_dim: usize,
        seed: Option<u64>,
    ) -> EmbeddingTable<Matrix<Node>, Node> {
        let mut rng = if let Some(seed_n) = seed {
            ChaCha8Rng::seed_from_u64(seed_n)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let mut table = Vec::new();
        let norm = Normal::new(0.0, 1.0).unwrap();
        for _ in 0..vocab_size {
            let mut items = Vec::new();
            for _ in 0..emb_dim {
                items.push(Node::from_f64(norm.sample(&mut rng)))
            }
            table.push(items);
        }
        EmbeddingTable {
            table: Matrix::from_vec(table),
            phantom_tensor_primitive: PhantomData,
        }
    }

    // /// Retrieve a set of embeddings
    // pub fn embs(&self, x: Matrix<Node>) -> Result<Matrix<Node>, MatrixError> {
    //     if x.shape()[1] != self.table.shape()[0] {
    //         return Err(MatrixError::DimMismatch(x.shape(), self.table.shape()));
    //     }
    //     x.matmul(&self.table)
    // }

    // pub fn at(&self, idxs: (usize, usize)) -> Option<&Node> {
    //     self.table.at(vec![idxs.0, idxs.1])
    // }

    // pub fn params(&self) -> Vec<CellPtr> {
    //     let mut v: Vec<CellPtr> = Vec::new();
    //     v.append(
    //         self.table
    //             .clone()
    //             .into_iter()
    //             .map(|node| node.leaf().expect("all layer params are leaves").clone())
    //             .collect::<Vec<CellPtr>>()
    //             .as_mut(),
    //     );
    //     v
    // }
}

pub enum EmbeddingTableError {
    IndexError(MatrixError),
}

impl Display for EmbeddingTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingTableError::IndexError(e) => write!(
                f,
                "Second dimension of table indexing matrix x must match table vocab size: {e}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn retrieve_embs() {
    //     let seed = 1;
    //     let tab = EmbeddingTable::new(3, 1, Some(seed));

    //     // (number of embeds, vocab_size)
    //     // set of 2 one hot vectors
    //     let mut x = Node::fill_matrix_f64((2, 3), 0.0);
    //     x.at_mut(vec![0, 0]).unwrap().leaf().unwrap().add_data(1.0);
    //     x.at_mut(vec![1, 1]).unwrap().leaf().unwrap().add_data(1.0);
    //     println!("{}", x);

    //     let embs = tab.embs(x).unwrap();
    //     println!("{}", embs);
    //     println!("{}", tab.table);
    //     // first embed in set
    //     assert_eq!(
    //         embs.at(vec![0, 0]).unwrap().resolve(),
    //         tab.at((0, 0)).unwrap().resolve()
    //     );
    //     // second embed in set
    //     assert_eq!(
    //         embs.at(vec![1, 0]).unwrap().resolve(),
    //         tab.at((1, 0)).unwrap().resolve()
    //     );
    // }
}
