use interfaces::{MathPrimitive, MathTensor, Tensor};
use matrix_library::Matrix;
use micrograd::node::Node;

pub fn softmax(x: &Vec<Vec<f64>>, dim: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; x[0].len()]; x.len()];

    if dim == 1 {
        for (j, row) in x.iter().enumerate() {
            let mut row_exp_sum = 0.0;
            for item in row.iter() {
                row_exp_sum += item.exp();
            }
            for (i, item) in row.iter().enumerate() {
                out[j][i] = item.exp() / row_exp_sum;
            }
        }
    } else {
        panic!("softmax across dim=0 not supported yet")
    }

    out
}

pub fn class_cross_entropy<MT: MathTensor<P>, T: Tensor<usize>, P: MathPrimitive>(
    x: &Vec<MT>,
    y_idx: &T,
) -> P {
    // implementation optimised for discrete target class index as the ground truth
    // input x are logits
    // - x_i + log( sum n->N[ e^x_n ] )

    // e was [2,4]
    // [[1,3],[5]]

    // x.shape = (B,T,C)
    let (batch_size, seq_len) = (y_idx.shape()[0], y_idx.shape()[1]);

    let mut loss = P::from_f64(0.0);

    for b in 0..batch_size {
        for t in 0..seq_len {
            // println!("{}", x[b]);
            let mut acc = x[b].at(vec![t, 0]).unwrap().clone().exp();
            // println!("{}", acc);
            for i in 1..x[b].shape()[1] {
                acc += x[b].at(vec![t, i]).unwrap().clone().exp();
            }
            // println!("{}", acc);
            loss += P::from_f64(-1.0)
                * x[b]
                    .at(vec![t, *y_idx.at(vec![b, t]).unwrap()])
                    .unwrap()
                    .clone()
                + acc.ln()
        }
    }

    loss

    // let mut acc = Value::new(E).pow_val(&Value::new(x[0]));
    // for i in 1..x.len() {
    //     acc = acc + Value::new(E).pow_val(&Value::new(x[i]));
    // }
    // let mut loss = &Value::new(-1.0) * &Value::new(x[y_idx]) + acc.ln();

    // let leaves = loss.backward(1.0);
    // // graph leaves corresonding to inputs occur at odd positions in the expression above
    // // leaf at index 1 references the input at position y_idx
    // let mut input_grads = Vec::new();
    // for (i, leaf) in leaves.iter().enumerate() {
    //     if i % 2 != 0 && i != 1 {
    //         input_grads.push(leaf.grad)
    //     }
    // }
    // input_grads[y_idx] += leaves[1].grad;
    // (loss, input_grads)
}
