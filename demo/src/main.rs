use transformer::model::Transformer;
fn main() {
    let vocab_size = 10;
    let n_embd = 32;
    let block_size = 8;
    let n_layers = 3;
    let n_heads = 3;
    let transformer = Transformer::new(vocab_size,n_embd,block_size,n_layers,n_heads);

}
