use crate::serial::{Layer, LayerError};
use matrix_library::Matrix;
use micrograd::{CellPtr, Node, Value};
use rand::{distributions::Distribution, thread_rng, Rng};
use statrs::distribution::Normal;

pub struct DenseLayer {
    pub i_size: usize,
    pub o_size: usize,
    w: Matrix<CellPtr>,
    b: Matrix<CellPtr>,
    // w: Vec<Value>,
    // b: Vec<Value>,
}

impl Layer for DenseLayer {
    fn forward(&self, x: Matrix<CellPtr>) -> Matrix<CellPtr> {
        let res = Matrix::matmul(&self, b)



        let mut res = Vec::new();
        for o in 0..self.o_size {
            let mut tmp_trees = Vec::new();
            for i in 0..self.i_size {
                tmp_trees.push(&Value::new(x[i]) * &(self.w[(o * self.i_size) + i]));
            }
            // t0 = x0*w00 + x1*w01 + x2*w02 ... + b0
            let mut tree_sum = tmp_trees.remove(0) + tmp_trees.remove(0);
            if tmp_trees.len() > 0 {
                for _ in 0..tmp_trees.len() {
                    tree_sum = tree_sum + tmp_trees.remove(0);
                }
            }
            let t = tree_sum + &(self.b[o]);
            res.push(t.resolve());
            self.tree[o] = Some(t);
        }
        res
    }
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> DenseLayer {
        let mut rng = thread_rng();
        let norm = Normal::new(0.0, (2.0 / (input_size as f64)).sqrt()).unwrap();
        let mut w: Vec<Value> = Vec::with_capacity(input_size * output_size);
        let mut tree: Vec<Option<Node>> = Vec::with_capacity(input_size * output_size);
        let mut b = Vec::with_capacity(output_size);
        for _ in 0..(input_size * output_size) {
            w.push(Value::new(norm.sample(&mut rng)));
        }
        for _ in 0..output_size {
            b.push(Value::new(0.0));
            tree.push(None);
        }
        DenseLayer {
            i_size: input_size,
            o_size: output_size,
            w,
            b,
            tree,
            input_grad: vec![0.0; input_size],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_and_backward() {
        let mut layer = DenseLayer::new(2, 3);
        let out = layer.forward(vec![2.0, 3.0]);

        //calc out manually
        assert_eq!(
            out,
            vec![
                (layer.w[0].data * 2.0) + (layer.w[1].data * 3.0) + layer.b[0].data,
                (layer.w[2].data * 2.0) + (layer.w[3].data * 3.0) + layer.b[1].data,
                (layer.w[4].data * 2.0) + (layer.w[5].data * 3.0) + layer.b[2].data,
            ]
        );

        if let Some(ref t) = layer.tree[0] {
            println!("\n-------------------------------------\n Printing graph for first element in dense_layer output");
            println!("\n y0 = x0 * w00 + x1 * w01 + b0");
            println!("{t}\n-------------------------------------\n");
        }

        layer
            .backward(vec![1.0, 2.0, 3.0])
            .expect("tree should be set by call to foreward");
        assert_eq!(layer.w[0].grad, 2.0 * 1.0);
        assert_eq!(layer.w[1].grad, 3.0 * 1.0);
        assert_eq!(layer.w[2].grad, 2.0 * 2.0);
        assert_eq!(layer.w[3].grad, 3.0 * 2.0);
        assert_eq!(layer.w[4].grad, 2.0 * 3.0);
        assert_eq!(layer.w[5].grad, 3.0 * 3.0);

        assert_eq!(layer.b[0].grad, 1.0);
        assert_eq!(layer.b[1].grad, 2.0);
        assert_eq!(layer.b[2].grad, 3.0);
    }
}
