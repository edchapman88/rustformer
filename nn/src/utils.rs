use micrograd::{Node, Value};
use std::f64::consts::E;

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

pub fn class_cross_entropy(x: &Vec<f64>, y_idx: usize) -> (Node, Vec<f64>) {
    // implementation optimised for discrete target class index as the ground truth
    // input x are logits
    // - x_i + log( sum n->N[ e^x_n ] )

    // e was [2,4]
    // [[1,3],[5]]

    let mut acc = Value::new(E).pow_val(&Value::new(x[0]));
    for i in 1..x.len() {
        acc = acc + Value::new(E).pow_val(&Value::new(x[i]));
    }
    let mut loss = &Value::new(-1.0) * &Value::new(x[y_idx]) + acc.ln();

    let leaves = loss.backward(1.0);
    // graph leaves corresonding to inputs occur at odd positions in the expression above
    // leaf at index 1 references the input at position y_idx
    let mut input_grads = Vec::new();
    for (i, leaf) in leaves.iter().enumerate() {
        if i % 2 != 0 && i != 1 {
            input_grads.push(leaf.grad)
        }
    }
    input_grads[y_idx] += leaves[1].grad;
    (loss, input_grads)
}
