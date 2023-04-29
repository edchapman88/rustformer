

pub fn softmax(x:&Vec<Vec<f64>>, dim:usize) -> Vec<Vec<f64>> {
   let mut out:Vec<Vec<f64>> = vec![vec![0.0; x[0].len()]; x.len()];

    if dim == 1 {
        for (j,row) in x.iter().enumerate() {
            let mut row_exp_sum = 0.0;
            for item in row.iter() {
                row_exp_sum += item.exp();
            }
            for (i,item) in row.iter().enumerate() {
                out[j][i] = item.exp() / row_exp_sum;
            }
        }
    } else {
        panic!("softmax across dim=0 not supported yet")
    }

    out
}