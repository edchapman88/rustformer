use crate::dense_layer::{DenseLayer, LayerError};

pub struct Serial {
    layers: Vec<DenseLayer>
}

impl Serial {
    pub fn new(layer_sizes:Vec<(usize,usize)>) {
        let mut layers = Vec::new();
        for (i_size,o_size) in layer_sizes.iter() {
            layers.push(DenseLayer::new(*i_size,*o_size))
        }
    }
    pub fn forward(&mut self, mut x:Vec<f64>) -> Vec<f64> {
        for l in self.layers.iter_mut() {
            let y = l.forward(x);
            x = y;
        }
        x
    }
    pub fn backward(&mut self, mut out_grad:Vec<f64>) -> Result<(), LayerError> {
        for l in self.layers.iter_mut() {
            l.backward(out_grad)?;
            out_grad = Vec::from(l.input_grad.as_slice())
        }
        Ok(())
    }
}