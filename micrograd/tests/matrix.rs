// TODO refactor after restriction of output types for Add and Mul for Matrix
// Add<Output = T> for Matrix<T>

// use matrix_library::Matrix;
// use micrograd::cell_ptr::CellPtr;
// use std::collections::VecDeque;
// #[test]
// fn add_overflow() {
//     let a = Matrix::new(VecDeque::from([
//         VecDeque::from([
//             CellPtr::from_f64(1.0),
//             CellPtr::from_f64(2.0),
//             CellPtr::from_f64(3.0),
//         ]),
//         VecDeque::from([
//             CellPtr::from_f64(4.0),
//             CellPtr::from_f64(5.0),
//             CellPtr::from_f64(6.0),
//         ]),
//     ]));
//     let b = Matrix::new(VecDeque::from([
//         VecDeque::from([
//             CellPtr::from_f64(1.0),
//             CellPtr::from_f64(2.0),
//             CellPtr::from_f64(3.0),
//         ]),
//         VecDeque::from([
//             CellPtr::from_f64(4.0),
//             CellPtr::from_f64(5.0),
//             CellPtr::from_f64(6.0),
//         ]),
//     ]));
//     let ans = Matrix::new(VecDeque::from([
//         VecDeque::from([
//             CellPtr::from_f64(1.0) + CellPtr::from_f64(1.0),
//             CellPtr::from_f64(2.0) + CellPtr::from_f64(2.0),
//             CellPtr::from_f64(3.0) + CellPtr::from_f64(3.0),
//         ]),
//         VecDeque::from([
//             CellPtr::from_f64(4.0) + CellPtr::from_f64(4.0),
//             CellPtr::from_f64(5.0) + CellPtr::from_f64(5.0),
//             CellPtr::from_f64(6.0) + CellPtr::from_f64(6.0),
//         ]),
//     ]));
//     assert_eq!(a + b, ans);
// }

// #[test]
// fn mul_overflow() {
//     let a = Matrix::new(VecDeque::from([
//         VecDeque::from([
//             CellPtr::from_f64(1.0),
//             CellPtr::from_f64(2.0),
//             CellPtr::from_f64(3.0),
//         ]),
//         VecDeque::from([
//             CellPtr::from_f64(4.0),
//             CellPtr::from_f64(5.0),
//             CellPtr::from_f64(6.0),
//         ]),
//     ]));
//     let b = Matrix::new(VecDeque::from([
//         VecDeque::from([
//             CellPtr::from_f64(1.0),
//             CellPtr::from_f64(2.0),
//             CellPtr::from_f64(3.0),
//         ]),
//         VecDeque::from([
//             CellPtr::from_f64(4.0),
//             CellPtr::from_f64(5.0),
//             CellPtr::from_f64(6.0),
//         ]),
//     ]));
//     let ans = Matrix::new(VecDeque::from([
//         VecDeque::from([
//             CellPtr::from_f64(1.0) * CellPtr::from_f64(1.0),
//             CellPtr::from_f64(2.0) * CellPtr::from_f64(2.0),
//             CellPtr::from_f64(3.0) * CellPtr::from_f64(3.0),
//         ]),
//         VecDeque::from([
//             CellPtr::from_f64(4.0) * CellPtr::from_f64(4.0),
//             CellPtr::from_f64(5.0) * CellPtr::from_f64(5.0),
//             CellPtr::from_f64(6.0) * CellPtr::from_f64(6.0),
//         ]),
//     ]));
//     assert_eq!(a * b, ans);
// }
