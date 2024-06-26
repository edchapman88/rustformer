use matrix_library::{Matrix, MatrixError};
use micrograd::cell_ptr::CellPtr;
use micrograd::node::Node;
use nn::utils::class_cross_entropy;
use nn::{
    dense_layer::DenseLayer, embedding_table::EmbeddingTable, relu_layer::ReluLayer, serial::Layer,
};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct Transformer {
    token_emb: EmbeddingTable,
    pos_emb: EmbeddingTable,
    hidden_layers: Vec<Block>,
    lm_head: DenseLayer,
    block_size: usize,
    batch_size: usize,
    vocab_size: usize,
}

struct Block {
    self_attention: SelfAttention,
    linear1: DenseLayer,
    linear2: DenseLayer,
    act: ReluLayer,
}

struct SelfAttention {
    n_heads: usize,
    n_embd: usize,
    head_size: usize,
    key: DenseLayer,
    query: DenseLayer,
    value: DenseLayer,
}

impl Transformer {
    pub fn new(
        vocab_size: usize,
        n_embd: usize,
        block_size: usize,
        n_layers: usize,
        n_heads: usize,
        head_size: usize,
        batch_size: usize,
        seed: Option<u64>,
    ) -> Transformer {
        Transformer {
            token_emb: EmbeddingTable::new(vocab_size, n_embd, seed),
            pos_emb: EmbeddingTable::new(block_size, n_embd, seed),
            hidden_layers: (0..n_layers)
                .map(|_| Block::new(n_embd, n_heads, head_size, seed))
                .collect::<Vec<Block>>(),
            lm_head: DenseLayer::new(vocab_size, n_embd, seed),
            block_size,
            batch_size,
            vocab_size,
        }
    }

    // input x: (B,T) where each element is a usize vocab index
    pub fn forward(
        &mut self,
        x: &Matrix<usize>,
        y: Option<&Matrix<usize>>,
    ) -> Result<(Vec<Matrix<Node>>, Option<Node>), MatrixError> {
        // drop batch parallelisation
        let mut batch_out = Vec::new();
        for batch_idx in 0..x.shape().0 {
            // create matrix with each row being a one-hot encoding of the token id
            let mut one_hots = Node::fill_matrix_f64((x.shape().1, self.vocab_size), 0.0);
            for i in 0..x.shape().1 {
                one_hots
                    .at_mut((i, *x.at((batch_idx, i)).unwrap()))
                    .expect("token index out of range of vocab size")
                    .leaf()
                    .unwrap()
                    .add_data(1.0);
            }

            let mut pos_one_hots = Node::fill_matrix_f64((x.shape().1, self.block_size), 0.0);
            for i in 0..x.shape().1 {
                pos_one_hots
                    .at_mut((i, i))
                    .unwrap()
                    .leaf()
                    .unwrap()
                    .add_data(1.0);
            }
            let mut emb_vec = self.token_emb.embs(one_hots).unwrap(); // (T,C)
            emb_vec = emb_vec + self.pos_emb.embs(pos_one_hots).unwrap();

            for block in self.hidden_layers.iter() {
                emb_vec = block.forward(&emb_vec)?;
            }

            // emb_vec is still (T,C)
            // lm_head maps (T,C) -> (T,vocab_size)
            let logits = self.lm_head.forward(&emb_vec)?;

            batch_out.push(logits);
        }

        if let Some(targets) = y {
            let loss = class_cross_entropy(&batch_out, targets);
            return Ok((batch_out, Some(loss)));
        }

        Ok((batch_out, None))
    }

    pub fn generate(
        &mut self,
        mut idx: Vec<usize>,
        n_new_tokens: usize,
        seed: Option<u64>,
    ) -> Result<Vec<usize>, MatrixError> {
        let mut rng = if let Some(seed_n) = seed {
            ChaCha8Rng::seed_from_u64(seed_n)
        } else {
            ChaCha8Rng::from_entropy()
        };
        for _ in 0..n_new_tokens {
            //crop to block_size if context is too long
            let mut ctx_block = &idx[..];
            if self.block_size < idx.len() {
                ctx_block = &idx[idx.len() - self.block_size..idx.len()];
            }
            // insert batch dimension to match expected shape
            let (logits, _) = self.forward(&Matrix::from_vecs(vec![ctx_block.to_vec()]), None)?;

            // logits shape: ( 1, len(ctx_block), vocab_size )
            // take first and only element from batch
            let el = logits[0].clone(); // (T, vocab_size)

            // take logits from final position in the ctx_block dimension - that will be
            // the next token prediction given all of the context provided
            let logits_sm = el.softmax(1);

            let mut next_token_probs = Vec::new();
            for i in 0..el.shape().1 {
                next_token_probs.push(logits_sm.at((el.shape().0 - 1, i)).unwrap().resolve())
            }

            // generate prob distribution from softmax output
            let dist = WeightedIndex::new(&next_token_probs).unwrap();
            let pred = &dist.sample(&mut rng);

            idx.push(*pred);
        }
        Ok(idx)
    }

    pub fn params(&self) -> Vec<CellPtr> {
        let mut params = Vec::new();
        params.extend(self.token_emb.params());
        params.extend(self.pos_emb.params());
        params.extend(
            self.hidden_layers
                .iter()
                .map(|l| l.params())
                .reduce(|mut acc, mut params| {
                    acc.append(&mut params);
                    acc
                })
                .expect("expect at least one parameter in the model"),
        );

        params.extend(self.lm_head.params());
        params
    }
}

impl Block {
    pub fn new(n_embd: usize, n_heads: usize, head_size: usize, seed: Option<u64>) -> Block {
        Block {
            self_attention: SelfAttention::new(n_embd, n_heads, head_size, seed),
            linear1: DenseLayer::new(n_embd, 4 * n_embd, seed),
            linear2: DenseLayer::new(4 * n_embd, n_embd, seed),
            act: ReluLayer::new(),
        }
    }

    pub fn params(&self) -> Vec<CellPtr> {
        let mut params = Vec::new();
        params.extend(self.self_attention.params());
        params.extend(self.linear1.params());
        params.extend(self.linear2.params());
        params
    }

    // not parallelised yet, x: (T,C)
    fn forward(&self, x: &Matrix<Node>) -> Result<Matrix<Node>, MatrixError> {
        // allowing some bypass around the attention block and the mlp block:
        // let mut x_bypass = x.clone() + self.self_attention.forward(x)?;

        // self.linear2.forward(self.linear1.forward(&x_bypass)?)?;
        // x_bypass = x_bypass + self.mlp_block.forward(x_bypass)?;
        // TODO
        // x = (0..x.len()).map(|i| x[i] + self.self_attention.forward(x)[i]).collect();
        // x = (0..x.len()).map(|i| x[i] + self.act.forward(self.linear2.forward(self.linear1.forward(x)))[i]).collect();
        // x

        // pass one sample from the batch into the self attention layer
        // this sample contains a set of 'examples'
        self.self_attention.forward(&x)
        // (T,C) -> (T,head_size)
        // for now, leave out ff-nn's, which works if head_size = n_embd
        // add ff-nn's to .backward() when they are implemented here

        // how does the output of the self_attention feed into the feed
        // forward neural nets?
        // A: I think they do the projection from head_size to n_embd

        // // feed forward each token independently (C,)
        // let emb_prj = Vec::new();
        // for emb in emb_vec.iter() {
        //     let mut tmp_emb = block.linear1.forward(*emb);
        //     tmp_emb = block.linear2.forward(tmp_emb);
        //     tmp_emb = block.act.forward(tmp_emb);
        //     emb_prj.push(tmp_emb);
        // }
        // emb_vec = emb_prj;  // (T,C) feed emb_vec back through block loop
    }
}

impl SelfAttention {
    pub fn new(
        n_embd: usize,
        n_heads: usize,
        head_size: usize,
        seed: Option<u64>,
    ) -> SelfAttention {
        SelfAttention {
            n_heads,
            n_embd,
            head_size,
            key: DenseLayer::new(head_size, n_embd, seed),
            query: DenseLayer::new(head_size, n_embd, seed),
            value: DenseLayer::new(head_size, n_embd, seed),
        }
    }

    fn forward(&self, x: &Matrix<Node>) -> Result<Matrix<Node>, MatrixError> {
        // TODO:
        // currently not parallelised over the batch, so input (T,C)
        // each of the T tokens emit a key and query vector of head_size
        let k = self.key.forward(x)?;
        let q = self.query.forward(x)?;

        // k and q are now both (T, head_size)
        // dot product each key with each query, since there are T of each,
        // there are T * T combinations, and this operation can be performed with
        // a matrix multiplication (T, head_size) @ (head_size, T) -> (T,T)
        let mut wei =
            q.matmul(&k.transpose())? * Node::from_f64((self.head_size as f64).powf(-0.5));

        // note: order matters, because the jth row in wei must be a vector where
        // all of the elements are a dot product between the jth q vector and the
        // respective k vector.
        // in other words, the final row will be the token affinites when the final
        // token is querying.

        // for language modeling, the token affinites are masked so that a token can
        // only query tokens before it.
        for j in 0..wei.shape().1 {
            for i in 0..wei.shape().0 {
                if i > j {
                    *wei.at_mut((j, i)).unwrap() = Node::from_f64(f64::NEG_INFINITY);
                }
            }
        }

        // softmax across each row
        let wei_softmax = wei.softmax(1);

        // get value vector for each token in x
        let v = self.value.forward(x)?;

        // return (T,T) @ (T,head_size) -> (T,head_size)
        wei_softmax.matmul(&v)
    }

    pub fn params(&self) -> Vec<CellPtr> {
        let mut params = Vec::new();
        params.extend(self.key.params());
        params.extend(self.query.params());
        params.extend(self.value.params());
        params
    }
}
