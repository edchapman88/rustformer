use matrix_library::Matrix;
use micrograd::node::Node;
use nn::{
    dense_layer::DenseLayer, optim::OptimSGD, relu_layer::ReluLayer, serial,
    sigmoid_layer::SigmoidLayer,
};
use rand::{seq::SliceRandom, thread_rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[test]
fn model_inference() {
    let model = serial::Serial::new(vec![
        Box::new(DenseLayer::new(2, 5, None)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(1, 2, None)),
        Box::new(SigmoidLayer::new()),
    ]);
    let x = Matrix::fill((5, 1), Node::from_f64(1.0));
    let mut y = model.forward(&x).unwrap();
    y.at_mut((0, 0)).unwrap().backward(1.0);
    println!("{}", y.at((0, 0)).unwrap());
}

#[test]
fn model_learning() {
    let seed_n = 2;
    let mut rng = ChaCha8Rng::seed_from_u64(seed_n);
    let model = serial::Serial::new(vec![
        Box::new(DenseLayer::new(5, 2, Some(seed_n))),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(10, 5, Some(seed_n))),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(5, 10, Some(seed_n))),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(1, 5, Some(seed_n))),
        Box::new(SigmoidLayer::new()),
    ]);

    let params = model.params();

    let max_itr = 20000;
    let optim = OptimSGD::new(0.04, max_itr, params);

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
    for itr in 0..max_itr {
        let sample = xor.choose(&mut rng);
        let (xi, yi) = sample.unwrap();

        let y_pred = model.forward(xi).unwrap();

        let mut e = bce(
            yi.at((0, 0)).unwrap().clone(),
            y_pred.at((0, 0)).unwrap().clone(),
        );

        // also tested with mse for comparison
        // let mut e = mse(
        //     yi.at((0, 0)).unwrap().clone(),
        //     y_pred.at((0, 0)).unwrap().clone(),
        // );

        if itr % 80 == 0 {
            let e_val = e.resolve();
            println!("{:?}", e_val);
        }

        e.zero_grad();
        e.backward(1.0);
        optim.update(itr);
    }

    for (xi, yi) in xor.iter() {
        let y_true = yi.at((0, 0)).unwrap();
        println!(
            "{:?} -> true: {}",
            (
                xi.at((0, 0)).unwrap().resolve(),
                xi.at((1, 0)).unwrap().resolve()
            ),
            y_true.resolve()
        );
        let y_pred = model.forward(&xi).unwrap().at((0, 0)).unwrap().clone();
        println!(
            "{:?} -> pred: {}",
            (
                xi.at((0, 0)).unwrap().resolve(),
                xi.at((1, 0)).unwrap().resolve()
            ),
            y_pred.resolve()
        );

        // require accuracy below a tolerance
        assert!(mse(y_true.clone(), y_pred).resolve() < 0.01);
    }
}

fn bce(y: Node, y_pred: Node) -> Node {
    // -1 * [ y * (y_pred + 0.0001).ln()    +    (1 - y) * (1 - (y_pred - 0.0001)).ln() ]
    Node::from_f64(-1.0)
        * (y.clone() * (y_pred.clone() + Node::from_f64(0.0000001)).ln()
            + (Node::from_f64(1.0) - y)
                * (Node::from_f64(1.0) - (y_pred - Node::from_f64(0.0000001))).ln())
}

fn mse(y: Node, y_pred: Node) -> Node {
    (y - y_pred).pow(Node::from_f64(2.0))
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
