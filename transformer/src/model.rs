use matrix_library::{Matrix, MatrixError};
use micrograd::node::Node;
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
    logits_loss_grad: Vec<Vec<Vec<f64>>>,
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
            logits_loss_grad: vec![vec![vec![0.0; vocab_size]; block_size]; batch_size],
        }
    }

    // input x: (B,T) where each element is a usize vocab index
    pub fn forward(
        &mut self,
        x: &Matrix<usize>,
        y: Option<&Matrix<usize>>,
    ) -> Result<(Vec<Matrix<Node>>, Option<Vec<Node>>), MatrixError> {
        // let mut batch_loss = Node::placeHolder();

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

            // for (example_idx, example) in emb_vec.iter().enumerate() {
            //     // each example has used a different number of tokens in it's context
            //     // but the dimensionality of each example is still (C,)
            //     // because the attention mechanism compressed the larger contexts into
            //     // a vector of the same length C

            //     // lm_head outputs (vocab_size,)
            //     // TODO: refactor loop with an iterator than consumes, to avoid clone() below
            //     let logits = self.lm_head.forward(example.clone());
            //     if let Some(targets) = y {
            //         // combined form of softmax and cross-entropy converts logits -> loss
            //         let input_grads: Vec<f64>;
            //         if batch_idx == 0 && example_idx == 0 {
            //             (batch_loss, input_grads) =
            //                 class_cross_entropy(&logits, targets[batch_idx][example_idx]);
            //         } else {
            //             let loss: Node;
            //             (loss, input_grads) =
            //                 class_cross_entropy(&logits, targets[batch_idx][example_idx]);
            //             batch_loss = batch_loss + loss;
            //         }
            //         self.logits_loss_grad[batch_idx][example_idx] = input_grads;
            //     }
            //     batch_item.push(logits);
            // }

            // rebuild batch of independent items
            // batch_out.push(batch_item); // (B,T,vocab_size)
        }
        // // batch loss on 'seen' training data
        // if let Some(_) = y {
        //     (batch_out, Some(batch_loss.resolve()))
        // } else {
        //     (batch_out, None)
        // }
        Ok((batch_out, None))
    }

    // pub fn backward(&mut self) {
    //     let mut loss_grad_sum = vec![0.0; self.vocab_size];
    //     for examples_loss_grads in self.logits_loss_grad.iter() {
    //         for loss_grads in examples_loss_grads.iter() {
    //             // lm_head nn has accrued loss from each consequtive example
    //             // since we haven't got the whole computation graph stored for each of the
    //             // model components, but instead only the most recent example in the most
    //             // recent item in the batch; we will backprop the sum of the errors accrued
    //             // for each of the tokens in the vocab - which is the 3rd dim of logits_loss_grad

    //             // TODO: this might be great regularisation, or it might dilute the learning signal
    //             // to much by the time the gradients reach the early layers
    //             loss_grad_sum = loss_grad_sum
    //                 .iter()
    //                 .zip(loss_grads.iter())
    //                 .map(|(&sum, &loss)| sum + loss)
    //                 .collect();
    //         }
    //     }
    //     self.lm_head.backward(loss_grad_sum);
    //     let mut out_grad = Vec::from(self.lm_head.get_input_grad());
    //     for block in self.hidden_layers.iter_mut().rev() {
    //         // TODO
    //         // block.backward(out_grad);
    //     }
    // }

    // pub fn zero_grad(&mut self) {
    //     self.lm_head.zero_grad();
    //     for block in self.hidden_layers.iter_mut() {
    //         block.zero_grad();
    //     }
    // }

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

    // fn zero_grad(&mut self) {
    //     self.self_attention.zero_grad();
    //     self.linear1.zero_grad();
    //     self.linear2.zero_grad();
    // }

    // fn backward(&mut self, mut out_grad: Vec<f64>) -> Result<(), LayerError> {
    //     // TODO: include ff-nn backprop when they are implemented in .forward()
    //     self.self_attention.backward(out_grad)?;
    //     Ok(())
    // }

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

    // fn zero_grad(&mut self) {
    //     self.key.zero_grad();
    //     self.query.zero_grad();
    //     self.value.zero_grad();
    // }

    // fn backward(&mut self, mut out_grad: Vec<f64>) -> Result<(), LayerError> {
    //     panic!("not implimented");
    //     Ok(())
    // }

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
        let mut wei = q.matmul(&k.transpose())?;
        // * Node::from_f64(self.head_size as f64).pow(Node::from_f64(-0.5));

        // note: order matters, because the jth row in wei must be a vector where
        // all of the elements are a dot product between the jth q vector and the
        // respective k vector.
        // in other words, the final row will be the token affinites when the final
        // token is querying.

        // for language modeling, the token affinites are masked so that a token can
        // only query tokens before it.
        let mut wei_mask: Vec<Vec<_>> = Vec::new();
        for j in 0..wei.shape().1 {
            wei_mask.push(Vec::new());
            for i in 0..wei.shape().0 {
                if i > j {
                    wei_mask[j].push(Node::from_f64(f64::NEG_INFINITY));
                } else {
                    wei_mask[j].push(Node::from_f64(1.0));
                }
            }
        }
        wei = wei * Matrix::from_vecs(wei_mask);

        // softmax across each row
        let wei_softmax = wei.softmax(1);

        // get value vector for each token in x
        let v = self.value.forward(x)?;

        // return (T,T) @ (T,head_size) -> (T,head_size)
        wei_softmax.matmul(&v)
    }
}
