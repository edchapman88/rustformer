use crate::serial::Serial;

pub struct OptimSGD {
    l_rate: f64,
}

impl OptimSGD {
    pub fn new(l_rate: f64) -> OptimSGD {
        OptimSGD { l_rate, }
    }
    pub fn update(&self, model: &mut Serial) {
        model.update(self.l_rate);
    }
}