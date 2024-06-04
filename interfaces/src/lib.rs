use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, Mul, MulAssign},
    process::Output,
};

pub trait Exp {
    fn exp(self) -> Self;
}

pub trait Pow<Exponent = Self> {
    fn pow(self, exp: Exponent) -> Self;
}

pub trait Ln {
    fn ln(self) -> Self;
}

pub trait LinearLayer<T, P>: DLModule<T, P>
where
    T: Tensor<P>,
    P: Primitive,
{
}

pub trait ActivationLayer<T, P>: DLModule<T, P>
where
    T: Tensor<P>,
    P: Primitive,
{
}

pub trait EmbeddingLayer<T, P>: DLModule<T, P>
where
    T: Tensor<P>,
    P: Primitive,
{
}

/// Deep Learning Module, for a Tensor object T and it's primitive P.
pub trait DLModule<T, P>
where
    T: Tensor<P>,
    P: Primitive,
{
    type DLModuleError: Debug
        + From<<T as Tensor<P>>::TensorError>
        + Into<<T as Tensor<P>>::TensorError>;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError>;

    fn params(&self) -> Vec<P>;
}

pub trait Tensor<P>:
Debug
+ Clone
+ Sized
+ Iterator<Item = P>
+ Add<Output = Self>
+ Add<P, Output = Self>
// + AddAssign
+ Mul<Output = Self>
+ Mul<P, Output = Self>
// + MulAssign
where P: Primitive {
    type TensorError: Debug;

    // fn from_vec(shape: Vec<usize>, data: Vec<P>) -> Self;
    fn from_vec(data: Vec<Vec<P>>) -> Self;

    /// Fill a matrix by repeatedly cloning the provided element.
    /// Note: the bahaviour might be unexpected if the provided element clones "by reference".
    fn fill_with_clone(shape: Vec<usize>, element: P) -> Self;

    fn shape(&self) -> Vec<usize>;

    fn transpose(self) -> Self;

    fn at(&self, idxs: Vec<usize>) -> Option<&P>;

    fn at_mut(&mut self, idxs: Vec<usize>) -> Option<&mut P>;

    fn matmul(&self, other: &Self) -> Result<Self, Self::TensorError>;

    /// Sum across one or more dimensions (eg. row-wise sum for a 2D matrix resulting in a "column vector")
    fn dim_sum(&self, dim: Vec<usize>) -> Self;
}

pub trait MathTensor<P>: Tensor<P> + Exp + Pow<P>
where
    P: MathPrimitive,
{
    /// Softmax across one dimension, leaving shape unchanged
    fn softmax(&self, dim: usize) -> Self;

    /// Fill a tensor with calls to `MathPrimitive::from_f64`
    /// Note: May provide different behaviour to `Tensor::fill_with_clone` (eg. by creating "new"
    /// primitives rather than cloning existing primitives).
    fn fill_from_f64(shape: Vec<usize>, data: f64) -> Self;
}

/// Computation Graph Node
pub trait Primitive:
    Debug
    + Clone
    + Display
    + Add<Output = Self>
    + AddAssign
    + Mul<Output = Self>
    // + MulAssign
    + Div<Output = Self>
{
}

pub trait MathPrimitive: Primitive + Exp + Pow + Ln {
    fn from_f64(data: f64) -> Self;

    /// "Resolve" the scalar primitive as f64.
    fn as_f64(&self) -> f64;
}

impl Primitive for usize {}
impl Primitive for u32 {}
impl Primitive for u16 {}
impl Primitive for i32 {}
impl Primitive for f64 {}

impl MathPrimitive for f64 {
    fn from_f64(data: f64) -> Self {
        data
    }

    fn as_f64(&self) -> f64 {
        *self
    }
}

impl Exp for f64 {
    fn exp(self) -> Self {
        self.exp()
    }
}

impl Pow for f64 {
    fn pow(self, exp: Self) -> Self {
        self.powf(exp)
    }
}

impl Ln for f64 {
    fn ln(self) -> Self {
        self.ln()
    }
}
