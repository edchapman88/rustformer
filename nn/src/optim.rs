use crate::serial::Serial;

pub struct OptimSGD {
    l_rate: f64,
    max_itr: usize
}

impl OptimSGD {
    pub fn new(l_rate: f64, max_itr: usize) -> OptimSGD {
        OptimSGD { l_rate, max_itr }
    }
    pub fn update(&self, model: &mut Serial, itr: usize) {
        if itr > self.max_itr.saturating_mul(3).saturating_div(4) {
            model.update(self.l_rate * 0.1);
        } else {
            model.update(self.l_rate);
        }
        
    }
}