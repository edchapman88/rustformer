use std::ops::{Add, Mul, Sub};
use autograd as ag;

#[derive(PartialEq, Debug)]
struct Value<'a> {
    data: i32,
    grad: i32,
    _op: Option<Operations<'a>>
}

#[derive(PartialEq, Debug)]
enum Operations<'a> {
    Add(Box<(&'a mut Value<'a>, &'a mut Value<'a>)>),
    Sub(Box<(Value<'a>,Value<'a>)>),
    Mul(Box<(Value<'a>, Value<'a>)>),
    Pow(Box<&'a mut Value<'a>>,u32)
}

impl Operations<'_> {
    fn _backward (&mut self, out_grad: i32) {
        match self {
            Operations::Add(bxd_tup) => {
                let (val1, val2) = &mut *(*bxd_tup);
                val1.grad += out_grad;
                val2.grad += out_grad;

                val1.backward();
                val2.backward();
            },
            Operations::Sub(bxd_tup) => {

            },
            Operations::Mul(bxd_tup) => {
                let (val1, val2) = &mut *(*bxd_tup);
                val1.grad += val2.data * out_grad;
                val2.grad += val1.data * out_grad;

                val1.backward();
                val2.backward();
            },
            Operations::Pow(bxd_val, exp) => {
                let val = &mut *(*bxd_val);
                val.grad += (*exp as i32 * val.data.pow(*exp-1_u32)) * out_grad;
            }
        }
    }
}

impl Value<'_> {
    fn new(data: i32, op: Option<Operations>) -> Value {
        Value { data, grad: 0, _op:op }
    }

    fn pow(self, exp: u32) -> Self {
        Value::new(self.data.pow(exp),
                    Some(Operations::Pow(Box::new(&mut self),exp)))
    }

    fn backward(&mut self) {
        if let Some(ref mut op) = self._op {
            op._backward(self.grad);
        }
    }

}

// impl Clone for Value {
//     fn clone(&self) -> Self {
//         let mut prev: Option<Box<(Value,Value)>> = None;
//         if let Some(boxed_tuple) = &self._prev {
//             prev = Some(Box::new(*boxed_tuple.clone()));
//         }
//         if let Some(op) = &self._op {
//            Value { data: self.data, grad: self.grad, _prev: prev, _op: Some(*op) } 
//         } else {
//             Value { data: self.data, grad: self.grad, _prev: prev, _op: None } 
//         }
        
//     }
// }

impl Add for Value<'_> {
    type Output = Self;
    fn add(mut self, mut rhs: Self) -> Self::Output {
        Value::new(
            self.data + rhs.data,   // data
            Some(Operations::Add(Box::new((&mut self, &mut rhs)))))  // operation
    }
}

impl Sub for Value<'_> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Value::new(
            self.data - rhs.data,   // data
            Some(Operations::Sub(Box::new((self,rhs)))))  // operation
    }
}

impl Mul for Value<'_> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Value::new(
            self.data * rhs.data,   // data
            Some(Operations::Mul(Box::new((self, rhs)))))  // operation
    }
}




#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn add_overflow() {
    //     let val1 = Value::new(3_i32, None);
    //     let val2 = Value::new(5_i32, None);
    //     assert_eq!(val1.clone() + val2.clone(), Value { data: 8_i32,
    //                                     grad: 0_i32,
    //                                     _op: Some(Operations::Add(Box::new((val1,val2))))
    //                                 })
    // }
    // #[test]
    // fn sub_overflow() {
    //     let val1 = Value::new(3_i32, None);
    //     let val2 = Value::new(5_i32, None);
    //     assert_eq!(val1.clone() - val2.clone(), Value { data: -2_i32,
    //                                     grad: 0_i32,
    //                                     _op: Some(Operations::Sub(Box::new((val1,val2))))
    //                                 })
    // }
    // #[test]
    // fn mull_overflow() {
    //     let val1 = Value::new(3_i32, None);
    //     let val2 = Value::new(5_i32, None);
    //     assert_eq!(val1.clone() * val2.clone(), Value { data: 15_i32,
    //                                     grad: 0_i32,
    //                                     _op: Some(Operations::Mul(Box::new((val1,val2))))
    //                                 })
    // }
    // #[test]
    // fn pow() {
    //     let val1 = Value::new(3_i32, None);
    //     // let val2 = Value::new(5_i32, None, None);
    //     assert_eq!(val1.clone().pow(2_u32), Value { data: 9_i32,
    //                                     grad: 0_i32,
    //                                     _op: Some(Operations::Pow(Box::new(val1),2_u32))
    //                                 })
    // }

    #[test]
    fn backward() {
        let a = Value::new(2, None);
        let b = Value::new(3, None);
        let c = Value::new(4, None);
        let mut y = (a + b) * c;
        y.grad = 2;
        y.backward();
        
        let new_a = Value { data: 2, grad: 8, _op: None};
        let new_b = Value { data: 3, grad: 8, _op: None};
        let new_a_b = Value { data: 5, grad: 8, _op: Some(Operations::Add(Box::new((&mut new_a, &mut new_b))))};
        let new_c = Value { data: 4, grad: 10, _op: None};
        assert_eq!(y, Value { data: 20, grad: 2, _op: Some(Operations::Mul(Box::new((new_a_b,new_c))))});
    }

    #[test]
    fn test_against_autograd() {
        let x = Value::new(2, None);
        let y = Value::new(3, None);
        // let c = Value::new(4, None);

        let mut z = Value::new(2,None) * x.pow(2) + Value::new(3,None)*y + Value::new(1,None);
        z.backward();

        println!("{:?}",y);
        println!("{:?}",x);
        println!("{:?}",z);

        ag::with(|g: &mut ag::Graph<_>| {
            let x = g.placeholder(&[]);
            let y = g.placeholder(&[]);
            let z = 2.*x*x + 3.*y + 1.;
        
            // dz/dy
            let gy = &g.grad(&[z], &[y])[0];
            println!("{:?}", gy.eval(&[]));   // => Ok(3.)
        
            // dz/dx (requires to fill the placeholder `x`)
            let gx = &g.grad(&[z], &[x])[0];
            let feed = ag::ndarray::arr0(2.);
            println!("{:?}", gx.eval(&[x.given(feed.view())]));  // => Ok(8.)
            // ddz/dx (differentiates `z` again)
            let ggx = &g.grad(&[gx], &[x])[0];
            println!("{:?}", ggx.eval(&[]));  // => Ok(4.)
        });

    }
}
