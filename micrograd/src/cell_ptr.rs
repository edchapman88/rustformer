use crate::value::Value;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    val: RefCell<Value>,
}

impl Cell {
    pub fn new(data: f64) -> Cell {
        Cell {
            val: RefCell::new(Value::new(data)),
        }
    }
}

#[derive(PartialEq, Debug, Serialize, Deserialize)]
pub struct CellPtr {
    ptr: Rc<Cell>,
}
impl CellPtr {
    pub fn new(data: f64) -> CellPtr {
        CellPtr {
            ptr: Rc::new(Cell::new(data)),
        }
    }
    pub fn resolve(&self) -> f64 {
        self.ptr.val.borrow().resolve()
    }
    pub fn data_ref(&self) -> f64 {
        self.ptr.val.borrow().resolve()
    }
    pub fn grad_ref(&self) -> f64 {
        self.ptr.val.borrow().grad
    }
    pub fn add_grad(&self, g: f64) {
        self.ptr.val.borrow_mut().grad += g
    }
    pub fn add_data(&self, delta: f64) {
        self.ptr.val.borrow_mut().data += delta
    }
    pub fn load_data(&self, data: f64, grad: f64) {
        *self.ptr.val.borrow_mut() = Value { data, grad };
    }
    pub fn zero_grad(&self) {
        self.ptr.val.borrow_mut().grad = 0.0
    }
}

impl Clone for CellPtr {
    fn clone(&self) -> Self {
        CellPtr {
            ptr: Rc::clone(&self.ptr),
        }
    }
}
