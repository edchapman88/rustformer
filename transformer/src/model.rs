use core::panic;

use matrix_library::Matrix;
use micrograd::{Node, Value};
use nn::serial::LayerError;
use nn::utils::class_cross_entropy;
use nn::{
    dense_layer::DenseLayer, embedding_table::EmbeddingTable, relu_layer::ReluLayer, serial::Layer,
};
use rand::prelude::Distribution;
use rand::{distributions::WeightedIndex, thread_rng, Rng};

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
    ) -> Transformer {
        Transformer {
            token_emb: EmbeddingTable::new((vocab_size, n_embd)),
            pos_emb: EmbeddingTable::new((block_size, n_embd)),
            hidden_layers: (0..n_layers)
                .map(|_| Block::new(n_embd, n_heads, head_size))
                .collect::<Vec<Block>>(),
            lm_head: DenseLayer::new(n_embd, vocab_size),
            block_size,
            batch_size,
            vocab_size,
            logits_loss_grad: vec![vec![vec![0.0; vocab_size]; block_size]; batch_size],
        }
    }

    // input x: (B,T) where each element is a usize vocab index
    pub fn forward(
        &mut self,
        x: &Vec<Vec<usize>>,
        y: Option<&Vec<Vec<usize>>>,
    ) -> (Vec<Vec<Vec<f64>>>, Option<f64>) {
        let emb_batch = self.token_emb.get_emb(&x); // (B,T,C)

        // emb_batch += self.pos_emb.get_emb(&x); // TODO: elemwise add (B,TC) + (B,T,C)
        let mut batch_out: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut batch_loss = Node::placeHolder();
        for (batch_idx, emb_vec) in emb_batch.iter().enumerate() {
            // drop batch parallelisation before linear layers, which dont support it yet
            // emb_vec: (T,C)
            let mut tmp_emb_vec = emb_vec.to_owned();
            // TODO: refactor with iter_mut() to avoid clone with to_owned()?
            for block in self.hidden_layers.iter_mut() {
                tmp_emb_vec = block.forward(&tmp_emb_vec);
            }

            // emb_vec is still (T,C)
            // feed each of the T examples through lm_head
            let mut batch_item: Vec<Vec<f64>> = Vec::new();
            for (example_idx, example) in tmp_emb_vec.iter().enumerate() {
                // each example has used a different number of tokens in it's context
                // but the dimensionality of each example is still (C,)
                // because the attention mechanism compressed the larger contexts into
                // a vector of the same length C

                // lm_head outputs (vocab_size,)
                // TODO: refactor loop with an iterator than consumes, to avoid clone() below
                let logits = self.lm_head.forward(example.clone());
                if let Some(targets) = y {
                    // combined form of softmax and cross-entropy converts logits -> loss
                    let input_grads: Vec<f64>;
                    if batch_idx == 0 && example_idx == 0 {
                        (batch_loss, input_grads) =
                            class_cross_entropy(&logits, targets[batch_idx][example_idx]);
                    } else {
                        let loss: Node;
                        (loss, input_grads) =
                            class_cross_entropy(&logits, targets[batch_idx][example_idx]);
                        batch_loss = batch_loss + loss;
                    }
                    self.logits_loss_grad[batch_idx][example_idx] = input_grads;
                }
                batch_item.push(logits);
            }

            // rebuild batch of independent items
            batch_out.push(batch_item); // (B,T,vocab_size)
        }
        // batch loss on 'seen' training data
        if let Some(_) = y {
            (batch_out, Some(batch_loss.resolve()))
        } else {
            (batch_out, None)
        }
    }

    pub fn backward(&mut self) {
        let mut loss_grad_sum = vec![0.0; self.vocab_size];
        for examples_loss_grads in self.logits_loss_grad.iter() {
            for loss_grads in examples_loss_grads.iter() {
                // lm_head nn has accrued loss from each consequtive example
                // since we haven't got the whole computation graph stored for each of the
                // model components, but instead only the most recent example in the most
                // recent item in the batch; we will backprop the sum of the errors accrued
                // for each of the tokens in the vocab - which is the 3rd dim of logits_loss_grad

                // TODO: this might be great regularisation, or it might dilute the learning signal
                // to much by the time the gradients reach the early layers
                loss_grad_sum = loss_grad_sum
                    .iter()
                    .zip(loss_grads.iter())
                    .map(|(&sum, &loss)| sum + loss)
                    .collect();
            }
        }
        self.lm_head.backward(loss_grad_sum);
        let mut out_grad = Vec::from(self.lm_head.get_input_grad());
        for block in self.hidden_layers.iter_mut().rev() {
            // TODO
            // block.backward(out_grad);
        }
    }

    pub fn zero_grad(&mut self) {
        self.lm_head.zero_grad();
        for block in self.hidden_layers.iter_mut() {
            block.zero_grad();
        }
    }

    pub fn generate(&mut self, mut idx: Vec<usize>, n_new_tokens: usize) -> Vec<usize> {
        let mut rng = thread_rng();
        for _ in 0..n_new_tokens {
            //crop to block_size if context is too long
            let mut ctx_block = &idx[..];
            if self.block_size < idx.len() {
                ctx_block = &idx[idx.len() - self.block_size..idx.len()];
            }
            // insert batch dimension to match expected shape
            let (logits, _) = self.forward(&vec![ctx_block.to_vec()], None);
            // ( 1, len(ctx_block), vocab_size )
            // take logits from final position in the ctx_block dimension - that will be
            // the next token prediction given all of the context provided

            let new_token_logits = logits[0][logits[0].len() - 1].to_owned();
            let probs = nn::utils::softmax(&vec![new_token_logits], 1)[0].to_owned();
            // generate prob distribution from softmax output
            let dist = WeightedIndex::new(&probs).unwrap();
            let pred = &dist.sample(&mut rng);
            idx.push(*pred);
        }
        idx
    }
}

impl Block {
    pub fn new(n_embd: usize, n_heads: usize, head_size: usize) -> Block {
        Block {
            self_attention: SelfAttention::new(n_embd, n_heads, head_size),
            linear1: DenseLayer::new(n_embd, 4 * n_embd),
            linear2: DenseLayer::new(4 * n_embd, n_embd),
            act: ReluLayer::new(),
        }
    }

    fn zero_grad(&mut self) {
        self.self_attention.zero_grad();
        self.linear1.zero_grad();
        self.linear2.zero_grad();
    }

    fn backward(&mut self, mut out_grad: Vec<f64>) -> Result<(), LayerError> {
        // TODO: include ff-nn backprop when they are implemented in .forward()
        self.self_attention.backward(out_grad)?;
        Ok(())
    }

    // not parallelised yet, x: (T,C)
    fn forward(&mut self, x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        // allowing some bypass around the attention block and the mlp block:
        // x = x + self_attemtion.forward(x)
        // x = x + mlp_block.forward(x)
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
    pub fn new(n_embd: usize, n_heads: usize, head_size: usize) -> SelfAttention {
        SelfAttention {
            n_heads,
            n_embd,
            head_size,
            key: DenseLayer::new(n_embd, head_size),
            query: DenseLayer::new(n_embd, head_size),
            value: DenseLayer::new(n_embd, head_size),
        }
    }

    fn zero_grad(&mut self) {
        self.key.zero_grad();
        self.query.zero_grad();
        self.value.zero_grad();
    }

    fn backward(&mut self, mut out_grad: Vec<f64>) -> Result<(), LayerError> {
        panic!("not implimented");
        Ok(())
    }

    fn forward(&mut self, x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        // TODO:
        // currently not parallelised over the batch, so input (T,C)
        // each of the T tokens emit a key and query vector of head_size
        let mut k = Vec::new();
        let mut q = Vec::new();
        for token in x.iter() {
            k.push(self.key.forward(token.to_owned())); // (C,) -> (head_size,)
            q.push(self.query.forward(token.to_owned()));
        }
        // k and q are now both (T, head_size)
        // dot product each key with each query, since there are T of each,
        // there are T * T combinations, and this operation can be performed with
        // a matrix multiplication (T, head_size) @ (head_size, T) -> (T,T)

        let q_mat = Matrix::new(q);
        let k_mat = Matrix::new(k);
        let wei: Vec<Vec<f64>> = q_mat.matmul(&k_mat.transpose()).unwrap().to_vec(); // (T, T)

        // note: order matters, because the jth row in wei must be a vector where
        // all of the elements are a dot product between the jth q vector and the
        // respective k vector.
        // in other words, the final row will be the token affinites when the final
        // token is querying.

        // for language modeling, the token affinites are masked so that a token can
        // only query tokens before it.
        let mut wei_masked: Vec<Vec<f64>> = vec![vec![0.0; wei.len()]; wei.len()];
        for (j, row) in wei.iter().enumerate() {
            for (i, item) in row.iter().enumerate() {
                if i > j {
                    wei_masked[j][i] = f64::NEG_INFINITY;
                } else {
                    wei_masked[j][i] = *item;
                }
            }
        }

        // softmax across each row
        let wei_softmax = nn::utils::softmax(&wei_masked, 1);

        // get value vector for each token in x
        let mut v: Vec<Vec<f64>> = Vec::new();
        for token in x.iter() {
            v.push(self.value.forward(token.to_owned()))
        }
        // return (T,T) @ (T,head_size) -> (T,head_size)
        let wei_mat = Matrix::new(wei_softmax);
        let v_mat = Matrix::new(v);
        wei_mat.matmul(&v_mat).unwrap().to_vec()
    }
}
