use crate::serial::{Layer,LayerError};

pub struct ReluLayer {
    x: Vec<f64>,
    pub input_grad: Vec<f64>
}

impl Layer for ReluLayer {
    fn forward(&mut self, x: Vec<f64>) -> Vec<f64> {
        let mut res = Vec::new();
        for el in &x {
            res.push(if *el > 0.0 {*el} else {0.0})
        }
        self.x = x;
        res
    }
    fn backward(&mut self, out_grad: Vec<f64>) -> Result<(), LayerError> {
        let mut input_grad = Vec::new();
        if self.x.len() == 0 {
            return Err(LayerError::MissingActivationInputs(String::from("
            Either missing a previous call to forward(), or the layer input has zero length")))
        }
        for (i,el) in self.x.iter().enumerate() {
            input_grad.push(out_grad[i] * if *el > 0.0 {1.0} else {0.0})
        }
        self.input_grad = input_grad;
        Ok(())
    }
    fn update(&mut self, l_rate: f64) {
        // No hyperparams to update in relu layer.
    }
    fn zero_grad(&mut self) {
        // No hyperparams to update in relu layer.
    }
    fn get_input_grad(&self) -> &[f64] {
        self.input_grad.as_slice()
    }
}

impl ReluLayer {
    pub fn new() -> ReluLayer {
        ReluLayer { x: vec![], input_grad: vec![] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_and_backward() {
        let mut layer = ReluLayer::new();
        let out = layer.forward(vec![-2.0,3.0,0.0,3.2,-0.5]);
        
        //calc out manually
        assert_eq!(out, vec![0.0,3.0,0.0,3.2,0.0]);
        
        layer.backward(vec![1.0,2.0,3.0,4.0,5.0]).expect("self.x should be set by call to foreward");
        assert_eq!(layer.get_input_grad(), &[0.0,2.0,0.0,4.0,0.0]);
        
    }
}
