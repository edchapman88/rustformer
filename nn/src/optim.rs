use micrograd::cell_ptr::CellPtr;
use serde::{Deserialize, Serialize};
use std::{fs, iter::zip};

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

    pub fn zero_grad(&self) {
        for p in self.params.iter() {
            p.zero_grad()
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

    pub fn load(&self, ckpt_path: &str) {
        let ckpt_params: Vec<CellPtr> =
            serde_json::from_str(&fs::read_to_string(ckpt_path).unwrap()).unwrap();
        for (cur, ckpt) in zip(&self.params, ckpt_params) {
            cur.load_data(ckpt.data_ref(), ckpt.grad_ref())
        }
    }

    pub fn save(&self, ckpt_path: &str) {
        fs::write(ckpt_path, serde_json::to_string(&self.params).unwrap()).unwrap();
    }
}
