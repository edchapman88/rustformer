#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
}

impl Value {
    pub fn new(data: f64) -> Value {
        Value { data, grad: 0.0 }
    }
    pub fn resolve(&self) -> f64 {
        self.data
    }
}
