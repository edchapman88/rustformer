use nn::{ embedding_table::EmbeddingTable, dense_layer::DenseLayer, relu_layer::ReluLayer, serial::Layer };


pub struct Transformer {
    token_emb: EmbeddingTable,
    pos_emb: EmbeddingTable,
    hidden_layers: Vec<Block>,
    lm_head: DenseLayer,
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
    key: DenseLayer,
    query: DenseLayer,
    value: DenseLayer
}

impl Transformer {
    pub fn new(vocab_size:usize, n_embd:usize, block_size:usize, n_layers:usize, n_heads:usize) -> Transformer {
        Transformer {
            token_emb: EmbeddingTable::new((vocab_size,n_embd)),
            pos_emb: EmbeddingTable::new((block_size, n_embd)),
            hidden_layers: (0..n_layers).map(|_| Block::new(n_embd,n_heads)).collect::<Vec<Block>>(),
            lm_head: DenseLayer::new(n_embd, vocab_size)
        }
    }
}

impl Block {
    pub fn new(n_embd:usize, n_heads:usize) -> Block {
        Block {
            self_attention: SelfAttention::new(n_embd,n_heads),
            linear1: DenseLayer::new(n_embd, 4*n_embd),
            linear2: DenseLayer::new(4*n_embd, n_embd),
            act: ReluLayer::new()
        }
    }
    fn forward(mut self, x:Vec<f64>) -> Vec<f64> {
        // allowing some bypass around the attention block and the mlp block:
        // x = x + self_attemtion.forward(x)
        // x = x + mlp_block.forward(x)
        // TODO
        // x = (0..x.len()).map(|i| x[i] + self.self_attention.forward(x)[i]).collect();
        // x = (0..x.len()).map(|i| x[i] + self.act.forward(self.linear2.forward(self.linear1.forward(x)))[i]).collect();
        // x
        self.act.forward(self.linear2.forward(self.linear1.forward(self.self_attention.forward(x))))
    }
}

impl SelfAttention {
    pub fn new(n_embd:usize,n_heads:usize) -> SelfAttention {
        SelfAttention { 
            n_heads,
            n_embd,
            key: DenseLayer::new(n_embd, n_embd),
            query: DenseLayer::new(n_embd, n_embd),
            value: DenseLayer::new(n_embd, n_embd),
        }
    }

    fn forward(self, x:Vec<f64>) -> Vec<f64> {
        // TODO
        vec![0.0; x.len()]
    }
}