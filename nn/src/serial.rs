use interfaces::{DLModule, Primitive, Tensor};

// pub trait Layer {
//     fn forward(&self, x: &Matrix<Node>) -> Result<Matrix<Node>, MatrixError>;
//     fn params(&self) -> Vec<CellPtr>;
// }

pub struct Serial<T, P>
where
    T: Tensor<P>,
    P: Primitive,
{
    pub layers: Vec<Box<dyn DLModule<T, P, DLModuleError = <T as Tensor<P>>::TensorError>>>,
}

impl<T, P> Serial<T, P>
where
    T: Tensor<P>,
    P: Primitive,
{
    pub fn new(
        layers: Vec<Box<dyn DLModule<T, P, DLModuleError = <T as Tensor<P>>::TensorError>>>,
    ) -> Serial<T, P> {
        Serial { layers }
    }
    pub fn forward(&self, x: &T) -> Result<T, <T as Tensor<P>>::TensorError> {
        let mut y = self
            .layers
            .first()
            .expect("at least one layer in Serial")
            .forward(x)?;
        let mut tmp = &y;
        for l in self.layers[1..].iter() {
            y = l.forward(tmp)?;
            tmp = &y;
        }
        Ok(y)
    }

    pub fn params(&self) -> Vec<P> {
        self.layers
            .iter()
            .map(|l| l.as_ref().params())
            .reduce(|mut acc, mut params| {
                acc.append(&mut params);
                acc
            })
            .expect("expect at least one parameter in the model")
    }
}
