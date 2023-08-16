use micrograd::cell_ptr::CellPtr;

pub struct OptimSGD {
    l_rate: f64,
    max_itr: usize,
    params: Vec<CellPtr>,
}

impl OptimSGD {
    pub fn new(l_rate: f64, max_itr: usize, params: Vec<CellPtr>) -> OptimSGD {
        OptimSGD {
            l_rate,
            max_itr,
            params,
        }
    }
    pub fn update(&self, itr: usize) {
        let mut l_rate = self.l_rate;
        if itr > self.max_itr.saturating_mul(3).saturating_div(4) {
            l_rate *= 0.1;
        }
        for p in self.params.iter() {
            p.add_data(-l_rate * p.grad_ref())
        }
    }
}
