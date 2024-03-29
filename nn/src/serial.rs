use crate::dense_layer::DenseLayer;

pub trait Layer {
    fn forward(&mut self, x: Vec<f64>) -> Vec<f64>;
    fn backward(&mut self, out_grad: Vec<f64>) -> Result<(), LayerError>;
    fn update(&mut self, l_rate: f64);
    fn zero_grad(&mut self);
    fn get_input_grad(&self) -> &[f64];
}
#[derive(Debug)]
pub enum LayerError {
    MissingTree(String),
    MissingActivationOutputs(String),
    MissingActivationInputs(String)
}

pub struct Serial {
    pub layers: Vec<Box<dyn Layer>>
}

impl Serial {
    pub fn new(layers:Vec<Box<dyn Layer>>) -> Serial {
        Serial { layers, }
    }
    pub fn forward(&mut self, x: &Vec<f64>) -> Vec<f64> {
        let mut tmp = x.to_vec();
        for l in self.layers.iter_mut() {
            let y = l.forward(tmp);
            tmp = y;
        }
        tmp
    }
    pub fn backward(&mut self, mut out_grad:Vec<f64>) -> Result<(), LayerError> {
        for l in self.layers.iter_mut().rev() {
            l.backward(out_grad)?;
            out_grad = Vec::from(l.get_input_grad())
        }
        Ok(())
    }
    pub fn update(&mut self, l_rate: f64) {
        for l in self.layers.iter_mut() {
            l.update(l_rate);
        }
    }
    pub fn zero_grad(&mut self) {
        for l in self.layers.iter_mut() {
            l.zero_grad();
        }
    }
}

