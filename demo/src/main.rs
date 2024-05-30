use matrix_library::Matrix;
use nn::optim::OptimSGD;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, HashSet};
use std::{fs, path::Path};

use transformer::model::Transformer;
fn main() {
    let seed = 1;
    // load in data and prepare
    let txt = fs::read_to_string(
        // Path::new(std::env::var("CARGO_WORKSPACE_DIR").unwrap().as_str())
        //     .join("data/tiny_shakespeare.txt"),
        "/Users/echapman/projects/rust-training/rustformer/data/tiny_shakespeare.txt",
    )
    .unwrap();

    let all_chars: Vec<char> = (&txt).chars().collect();

    // find set of all chars to build indexed vocabulary
    let mut set = HashSet::new();
    for c in (&all_chars).into_iter() {
        set.insert(*c);
    }
    let mut chars: Vec<char> = Vec::from_iter(set);
    chars.sort_by(|a, b| b.cmp(a));
    chars.reverse();
    // println!("{:?}",chars);
    let mut chartoi = HashMap::new();
    let mut itochar = HashMap::new();
    for (i, c) in chars.iter().enumerate() {
        chartoi.insert(*c, i);
        itochar.insert(i, *c);
    }

    // encode txt data into vocab indexes
    let mut charidxs = Vec::new();
    for c in all_chars.iter() {
        charidxs.push(*chartoi.get(c).unwrap())
    }

    let vocab_size = chars.len();
    // let vocab_size = 17;
    let n_embd = 5;
    let block_size = 8;
    let n_layers = 1;
    let n_heads = 1;
    let batch_size = 1;
    let n_itrs = 10;
    let head_size = n_embd; // for now, without ff-nn's to project head_size -> n_embd

    fn get_batch(
        data: &[usize],
        batch_size: usize,
        block_size: usize,
        seed: Option<u64>,
    ) -> (Matrix<usize>, Matrix<usize>) {
        // each x sample in the batch contains block_size (eg. = 4) items eg. [x1,x2,x3,x4]
        // and that sample contains 4 training samples or different length,
        // each corresponding to the 4 y samples returned with it:
        // x1 -> y1(=x2)
        // x1,x2 -> y2(=x3)
        // x1,x2,x3 -> y3(=x4)
        // x1,x2,x3,x4 -> y4(=x5)
        // this is a more space efficient way of storing samples (avoiding data duplication)
        let mut rng = if let Some(seed_n) = seed {
            ChaCha8Rng::seed_from_u64(seed_n)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..batch_size {
            let idx = rng.gen_range(0..(data.len() - block_size));
            x.push(data[idx..(idx + block_size)].to_vec());
            y.push(data[idx + 1..(idx + block_size + 1)].to_vec())
        }
        (Matrix::from_vecs(x), Matrix::from_vecs(y))
    }

    let mut transformer = Transformer::new(
        vocab_size,
        n_embd,
        block_size,
        n_layers,
        n_heads,
        head_size,
        batch_size,
        Some(seed),
    );

    let params = transformer.params();
    let optim = OptimSGD::new(0.04, n_itrs, params);
    // ___________________________________________
    // optim.load("./ckpt.json");
    // let new_idxs = transformer
    //     .generate([15, 16, 17, 18].to_vec(), 30, Some(seed))
    //     .unwrap();

    // let mut x_str = String::new();
    // for idx in new_idxs {
    //     x_str.push(itochar.get(&idx).unwrap().to_owned());
    // }
    // println!("{}", x_str);
    // ___________________________________________

    for itr in 0..n_itrs {
        let (x_batch, y_batch) = get_batch(&charidxs, batch_size, block_size, Some(seed));
        let (logits_batch, loss) = transformer.forward(&x_batch, Some(&y_batch)).unwrap();
        if let Some(l) = loss {
            optim.zero_grad();
            l.clear_all_caches();
            l.backward(1.0);
            optim.update(itr);
            println!("{}", l);
        }

        if itr % 3 == 0 {
            let new_idxs = transformer
                .generate([15, 16, 17, 18].to_vec(), 30, Some(seed))
                .unwrap();

            let mut x_str = String::new();
            for idx in new_idxs {
                x_str.push(itochar.get(&idx).unwrap().to_owned());
            }
            println!("itr == {itr}\n{}\n\n", x_str);
        }
    }

    optim.save("./ckpt.json");

    // let new_idxs = transformer
    //     .generate([13].to_vec(), 100, Some(seed))
    //     .unwrap();

    // let mut x_str = String::new();
    // for idx in new_idxs {
    //     x_str.push(itochar.get(&idx).unwrap().to_owned());
    // }
    // println!("{}", x_str);
}
