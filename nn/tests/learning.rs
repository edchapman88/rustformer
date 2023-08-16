use matrix_library::Matrix;
use micrograd::node::Node;
use nn::{
    dense_layer::DenseLayer, optim::OptimSGD, relu_layer::ReluLayer, serial,
    sigmoid_layer::SigmoidLayer,
};
use rand::{seq::SliceRandom, thread_rng};

#[test]
fn model_inference() {
    let model = serial::Serial::new(vec![
        Box::new(DenseLayer::new(2, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(1, 2)),
        Box::new(SigmoidLayer::new()),
    ]);
    let x = Matrix::fill((5, 1), Node::from_f64(1.0));
    let mut y = model.forward(&x).unwrap();
    y.at_mut((0, 0)).unwrap().backward(1.0);
    println!("{}", y.at((0, 0)).unwrap());
}

#[test]
fn model_learning() {
    let mut rng = thread_rng();
    let model = serial::Serial::new(vec![
        Box::new(DenseLayer::new(5, 2)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(5, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(5, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(3, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(1, 3)),
        Box::new(SigmoidLayer::new()),
    ]);

    let params = model.params();

    let max_itr = 10000;
    let optim = OptimSGD::new(0.5, max_itr, params);

    let xor: Vec<(Matrix<Node>, Matrix<Node>)> = vec![
        (
            Matrix::from_vecs(vec![vec![Node::from_f64(0.0)], vec![Node::from_f64(0.0)]]),
            Matrix::from_vecs(vec![vec![Node::from_f64(0.0)]]),
        ),
        (
            Matrix::from_vecs(vec![vec![Node::from_f64(1.0)], vec![Node::from_f64(0.0)]]),
            Matrix::from_vecs(vec![vec![Node::from_f64(1.0)]]),
        ),
        (
            Matrix::from_vecs(vec![vec![Node::from_f64(0.0)], vec![Node::from_f64(1.0)]]),
            Matrix::from_vecs(vec![vec![Node::from_f64(1.0)]]),
        ),
        (
            Matrix::from_vecs(vec![vec![Node::from_f64(1.0)], vec![Node::from_f64(1.0)]]),
            Matrix::from_vecs(vec![vec![Node::from_f64(0.0)]]),
        ),
    ];
    // let mut e_log = String::new();
    for itr in 0..max_itr {
        let sample = xor.choose(&mut rng);
        let (xi, yi) = sample.unwrap();

        let y_pred = model.forward(xi).unwrap();
        // println!("pred: {:?}", y_pred.at((0, 0)).unwrap().resolve());
        let mut e = bce(
            yi.at((0, 0)).unwrap().clone(),
            y_pred.at((0, 0)).unwrap().clone(),
        );
        if itr % 100 == 0 {
            let e_val = e.resolve();
            println!("{:?}", e_val);
            // if e_val > 0.01 {
            //     println!(
            //         "{},{}",
            //         xi.at((0, 0)).unwrap().resolve(),
            //         xi.at((1, 0)).unwrap().resolve()
            //     );
            // }
        }
        // println!("{:?}", e.resolve());
        // e_log += &e.resolve().to_string();
        // e_log += "\n";
        // println!(
        //     "{:?}",
        //     model
        //         .params()
        //         .iter()
        //         .map(|p| p.data_ref())
        //         .collect::<Vec<f64>>()[1]
        // );
        e.zero_grad();
        // println!("{}", e);
        // if itr % 10 == 0 {
        //     println!(
        //         "{:?}",
        //         model
        //             .params()
        //             .iter()
        //             .map(|p| p.grad_ref())
        //             .collect::<Vec<f64>>()[1]
        //     );
        // }
        // println!("___________________");
        e.backward(e.resolve());
        // println!("{}", e);
        // if itr % 10 == 0 {
        //     println!(
        //         "{:?}",
        //         model
        //             .params()
        //             .iter()
        //             .map(|p| p.grad_ref())
        //             .collect::<Vec<f64>>()[1]
        //     );
        // }
        // println!("_________nudge data__________");
        optim.update(itr);
        // println!("{}", e);
        // println!("_________next__________");
    }
    // println!("{}", e_log);
    for (xi, yi) in xor.iter() {
        println!(
            "{:?} -> true: {}",
            (
                xi.at((0, 0)).unwrap().resolve(),
                xi.at((1, 0)).unwrap().resolve()
            ),
            yi.at((0, 0)).unwrap().resolve()
        );
        println!(
            "{:?} -> pred: {}",
            (
                xi.at((0, 0)).unwrap().resolve(),
                xi.at((1, 0)).unwrap().resolve()
            ),
            model.forward(&xi).unwrap().at((0, 0)).unwrap().resolve()
        );
    }
}

fn bce(y: Node, y_pred: Node) -> Node {
    // -1 * [ y * (y_pred + 0.0001).ln()    +    (1 - y) * (1 - (y_pred - 0.0001)).ln() ]

    y.clone() * (y_pred.clone() + Node::from_f64(0.0000001)).ln()
        + (Node::from_f64(1.0) - y)
            * (Node::from_f64(1.0) - (y_pred - Node::from_f64(0.0000001))).ln()
}

// fn binary_x_entropy(y: &Vec<Vec<f64>>, y_pred: &Vec<Vec<f64>>) -> Vec<f64> {
//     // TODO: refactor to be auto-differentiable (see bce)
//     let batch_size = y.len();
//     let y_len = y[0].len();
//     let mut mean_loss = vec![0.0; y[0].len()];
//     for s in 0..batch_size {
//         for i in 0..y_len {
//             mean_loss[i] +=
//                 (y[s][i] * y_pred[s][i].ln() + (1.0 - y[s][i]) * (1.0 - y_pred[s][i]).ln())
//         }
//     }
//     for el in &mut mean_loss {
//         *el /= -1.0 * batch_size as f64;
//     }
//     mean_loss
// }
