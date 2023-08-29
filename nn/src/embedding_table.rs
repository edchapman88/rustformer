use std::fmt::Display;

use matrix_library::{Matrix, MatrixError};
use micrograd::node::Node;
use rand::{distributions::Distribution, rngs::ThreadRng};
use statrs::distribution::Normal;
pub struct EmbeddingTable {
    table: Matrix<Node>,
}

impl EmbeddingTable {
    pub fn new(vocab_size: usize, emb_dim: usize, mut rng: &mut ThreadRng) -> EmbeddingTable {
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
            table: Matrix::from_vecs(table),
        }
    }

    /// Retrieve a batch of embeddings
    pub fn embs(&self, x: Matrix<Node>) -> Result<Matrix<Node>, MatrixError> {
        if x.shape().1 != self.table.shape().0 {
            return Err(MatrixError::DimMismatch(x.shape(), self.table.shape()));
        }
        x.matmul(&self.table)
    }

    pub fn at(&self, idxs: (usize, usize)) -> Option<&Node> {
        self.table.at(idxs)
    }
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
    use rand::thread_rng;

    use super::*;

    #[test]
    fn retrieve_embs() {
        let mut rng = thread_rng();
        let tab = EmbeddingTable::new(3, 1, &mut rng);

        // (batch_size, vocab_size)
        // batch of 2 one hot vectors
        let mut x = Node::fill_matrix_f64((2, 3), 0.0);
        x.at_mut((0, 0)).unwrap().leaf().unwrap().add_data(1.0);
        x.at_mut((1, 1)).unwrap().leaf().unwrap().add_data(1.0);
        println!("{}", x);

        let embs = tab.embs(x).unwrap();
        println!("{}", embs);
        println!("{}", tab.table);
        // first batch element
        assert_eq!(
            embs.at((0, 0)).unwrap().resolve(),
            tab.at((0, 0)).unwrap().resolve()
        );
        // second batch element
        assert_eq!(
            embs.at((1, 0)).unwrap().resolve(),
            tab.at((1, 0)).unwrap().resolve()
        );
    }
}
