use std::cell::RefCell;
use std::rc::Rc;

use crate::value::Value;

#[derive(PartialEq, Debug, Clone)]
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

#[derive(PartialEq, Debug, Clone)]
pub struct CellPtr {
    ptr: Rc<Cell>,
}
impl CellPtr {
    pub fn new(cell: Rc<Cell>) -> CellPtr {
        CellPtr { ptr: cell }
    }
    pub fn from_f64(data: f64) -> CellPtr {
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
    pub fn clone(cell_ptr: &CellPtr) -> CellPtr {
        CellPtr {
            ptr: Rc::clone(&cell_ptr.ptr),
        }
    }
}
