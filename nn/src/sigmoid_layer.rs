use crate::serial::{Layer,LayerError};

pub struct SigmoidLayer {
    f_x: Vec<f64>,
    pub input_grad: Vec<f64>
}

impl Layer for SigmoidLayer {
    fn forward(&mut self, x: Vec<f64>) -> Vec<f64> {
        let mut res = Vec::new();
        for el in &x {
            res.push(1.0 / (1.0 + (-el).exp()))
        }
        self.f_x = res.clone();
        res
    }
    fn backward(&mut self, out_grad: Vec<f64>) -> Result<(), LayerError> {
        let mut input_grad = Vec::new();
        if self.f_x.len() == 0 {
            return Err(LayerError::MissingActivationOutputs(String::from("
            Either missing a previous call to forward(), or the layer output has zero length")))
        }
        for (i,fx) in self.f_x.iter().enumerate() {
            input_grad.push(out_grad[i] * (1.0 - fx)* fx)
        }
        self.input_grad = input_grad;
        Ok(())
    }
    fn update(&mut self, l_rate: f64) {
        // No hyperparams to update in sigmoid layer.
    }
    fn zero_grad(&mut self) {
        // No hyperparams to update in sigmoid layer.
    }
    fn get_input_grad(&self) -> &[f64] {
        self.input_grad.as_slice()
    }
}

impl SigmoidLayer {
    pub fn new() -> SigmoidLayer {
        SigmoidLayer { f_x: vec![], input_grad: vec![] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn forward_and_backward() {
    //     let mut layer = SigmoidLayer::new();
    //     let out = layer.forward(vec![-2.0,3.0,0.0]);
        
    //     //calc out manually
    //     assert_eq!(out, vec![0.11920292202211755,0.9525741268224334,0.5]);
        
    //     layer.backward(vec![1.0,2.0,3.0]).expect("self.x should be set by call to foreward");
    //     assert_eq!(layer.get_input_grad(), &[0.1049935854035065,0.090353319461824,0.75]);
        
    // }
}
