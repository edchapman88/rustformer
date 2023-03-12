
use micrograd::{ Value,Node };
use rand::{ thread_rng, Rng };
use crate::serial::{Layer,LayerError};

pub struct DenseLayer {
    pub i_size: usize,
    pub o_size: usize,
    w: Vec<Value>,
    b: Vec<Value>,
    tree: Vec<Option<Node>>,
    pub input_grad: Vec<f64>
}

impl Layer for DenseLayer {
    fn forward(&mut self, x: Vec<f64>) -> Vec<f64> {
        // generate a tree from the forward pass and assign to layer.tree
        // take a reference to the tree and resolve to a returned value
        let mut res = Vec::new();
        for o in 0..self.o_size {
            let mut tmp_trees = Vec::new();
            for i in 0..self.i_size {
                tmp_trees.push(&Value::new(x[i]) * &(self.w[(o*self.i_size)+i]));
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
            self.tree[o] = Some (t);
        }
        res
    }
    fn backward(&mut self, out_grad:Vec<f64>) -> Result<(),LayerError> {
        // if there is a tree, call backward on the tree
        // mutate the data in the layer with the grads returned from backward
        for o in 0..self.o_size {
            if let Some(ref mut tree) = self.tree[o] {
                let mut leaves:Vec<Value> = tree.backward(out_grad[o]);
                // grads are ordered in order of maths operations
                // eg. when o=0: [x0,w00,x1,w01, ... ,b0]
                for i in 0..self.i_size {
                    self.input_grad[i] += leaves.remove(0).grad;
                    self.w[(o*self.i_size)+i] = leaves.remove(0);
                }
                self.b[o] = leaves.remove(0);
            }
        }
        Ok(())
    }
    fn update(&mut self, l_rate: f64) {
        for mut wi in self.w.iter_mut() {
            wi.data -= l_rate * wi.grad;
        }
        for mut bi in self.b.iter_mut() {
            bi.data -= l_rate * bi.grad;
        }
    }
    fn zero_grad(&mut self) {
        for mut wi in self.w.iter_mut() {
            wi.grad = 0.0;
        }
        for mut bi in self.b.iter_mut() {
            bi.grad = 0.0;
        }
    }
    fn get_input_grad(&self) -> &[f64] {
        self.input_grad.as_slice()
    }
}

impl DenseLayer {
    pub fn new(input_size:usize, output_size:usize) -> DenseLayer {
        let mut rng = thread_rng();
        let mut w:Vec<Value> = Vec::with_capacity(input_size * output_size);
        let mut tree:Vec<Option<Node>> = Vec::with_capacity(input_size * output_size);
        let mut b = Vec::with_capacity(output_size);
        for _ in 0..(input_size * output_size) {
            w.push(Value::new(rng.gen()));
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
            input_grad: vec![0.0; input_size]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_and_backward() {
        let mut layer = DenseLayer::new(2,3);
        let out = layer.forward(vec![2.0,3.0]);
        
        //calc out manually
        assert_eq!(out, vec![(layer.w[0].data * 2.0) + (layer.w[1].data * 3.0) + layer.b[0].data,
                            (layer.w[2].data * 2.0) + (layer.w[3].data * 3.0) + layer.b[1].data,
                            (layer.w[4].data * 2.0) + (layer.w[5].data * 3.0) + layer.b[2].data,]);

        if let Some(ref t) = layer.tree[0] {
            println!("{}",t)
        }
        
        layer.backward(vec![1.0,2.0,3.0]).expect("tree should be set by call to foreward");
        assert_eq!(layer.w[0].grad, 2.0*1.0);
        assert_eq!(layer.w[1].grad, 3.0*1.0);
        assert_eq!(layer.w[2].grad, 2.0*2.0);
        assert_eq!(layer.w[3].grad, 3.0*2.0);
        assert_eq!(layer.w[4].grad, 2.0*3.0);
        assert_eq!(layer.w[5].grad, 3.0*3.0);

        assert_eq!(layer.b[0].grad, 1.0);
        assert_eq!(layer.b[1].grad, 2.0);
        assert_eq!(layer.b[2].grad, 3.0);
    }
}
