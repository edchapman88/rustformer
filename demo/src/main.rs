use std::{fs, path::Path};
use std::collections::{HashSet, HashMap};
use rand::prelude::Distribution;
use rand::{thread_rng, Rng, distributions::WeightedIndex};

use transformer::model::Transformer;
fn main() {

    // load in data and prepare
    let txt = fs::read_to_string(
                Path::new(std::env::var("CARGO_WORKSPACE_DIR").unwrap().as_str())
                .join("data/tiny_shakespeare.txt")).unwrap();

    let all_chars: Vec<char> = (&txt).chars().collect();

    // find set of all chars to build indexed vocabulary
    let mut set = HashSet::new();
    for c in (&all_chars).into_iter() {
        set.insert(*c);
    }
    let mut chars:Vec<char> = Vec::from_iter(set);
    chars.sort_by(|a, b| b.cmp(a));
    chars.reverse();
    // println!("{:?}",chars);
    let mut chartoi = HashMap::new();
    let mut itochar = HashMap::new();
    for (i,c) in chars.iter().enumerate() {
        chartoi.insert(*c, i);
        itochar.insert(i, *c);
    }

    // encode txt data into vocab indexes
    let mut charidxs = Vec::new();
    for c in all_chars[0..14].iter() {
        charidxs.push(*chartoi.get(c).unwrap())
    }

    let vocab_size = chars.len();
    let n_embd = 5;
    let block_size = 8;
    let n_layers = 1;
    let n_heads = 1;
    let batch_size = 1;
    let head_size = n_embd; // for now, without ff-nn's to project head_size -> n_embd

    fn get_batch(data:Vec<usize>, batch_size: usize, block_size: usize) -> (Vec<Vec<usize>>,Vec<Vec<usize>>) {
        // each x sample in the batch contains block_size (eg = 4) items eg. [x1,x2,x3,x4]
        // and that sample contains 4 training samples or different length, 
        // each corresponding to the 4 y samples returned with it:
        // x1 -> y1(=x2)
        // x1,x2 -> y2(=x3)
        // x1,x2,x3 -> y3(=x4)
        // x1,x2,x3,x4 -> y4(=x5)
        // this is a more space efficient way of storing samples (avoiding data duplication)
        let mut rng = rand::thread_rng();
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..batch_size {
            let idx = rng.gen_range(0..(data.len()-block_size));
            x.push(data[idx..(idx+block_size)].to_vec());
            y.push(data[idx+1..(idx+block_size+1)].to_vec())
        }
        (x,y)
    }

    println!("{:?}",chars);

    let (x_batch,y_batch) = get_batch(charidxs, batch_size, block_size);
    println!("x = {:?}",x_batch);
    println!("y = {:?}",y_batch);
    

    let mut transformer = Transformer::new(vocab_size,n_embd,block_size,n_layers,n_heads,head_size);
    let (logits_batch,loss) = transformer.forward(&x_batch,Some(&y_batch)); // (B,T,vocab_size)
    println!("shape of logits: {:?}", (logits_batch.len(),logits_batch[0].len(),logits_batch[0][0].len()));

    // drop batch paralleilisation
    for (logits,y) in logits_batch.iter().zip(y_batch.iter()) {
        let probs = nn::utils::softmax(&logits, 1); // (T,vocab_size)

    }
    
    
    // let new_idxs = transformer.generate([0].to_vec(), 100);

    // let mut x_str = String::new();
    // for idx in new_idxs {
    //     x_str.push(itochar.get(&idx).unwrap().to_owned());
    // }
    // println!("{:?}",x_str);
}
