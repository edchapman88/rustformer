use interfaces::Ln;
use interfaces::MathPrimitive;
use interfaces::Pow;
use interfaces::Tensor;
use matrix_library::Matrix;
use micrograd::node::Node;
use nn::{
    dense_layer::DenseLayer, optim::OptimSGD, relu_layer::ReluLayer, serial,
    sigmoid_layer::SigmoidLayer,
};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
#[test]
fn model_inference() {
    let model = serial::Serial::new(vec![
        Box::new(DenseLayer::new(2, 5, None)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(1, 2, None)),
        Box::new(SigmoidLayer::new()),
    ]);
    let x = Matrix::fill(vec![1, 5], Node::from_f64(1.0));
    let mut y = model.forward(&x).unwrap();
    y.at_mut(vec![0, 0]).unwrap().backward(1.0);
    println!("{}", y.at(vec![0, 0]).unwrap());
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
            Matrix::from_vec(vec![vec![Node::from_f64(0.0)], vec![Node::from_f64(0.0)]]),
            Matrix::from_vec(vec![vec![Node::from_f64(0.0)]]),
        ),
        (
            Matrix::from_vec(vec![vec![Node::from_f64(1.0)], vec![Node::from_f64(0.0)]]),
            Matrix::from_vec(vec![vec![Node::from_f64(1.0)]]),
        ),
        (
            Matrix::from_vec(vec![vec![Node::from_f64(0.0)], vec![Node::from_f64(1.0)]]),
            Matrix::from_vec(vec![vec![Node::from_f64(1.0)]]),
        ),
        (
            Matrix::from_vec(vec![vec![Node::from_f64(1.0)], vec![Node::from_f64(1.0)]]),
            Matrix::from_vec(vec![vec![Node::from_f64(0.0)]]),
        ),
    ];
    for itr in 0..max_itr {
        let sample = xor.choose(&mut rng);
        let (xi, yi) = sample.unwrap();

        let y_pred = model.forward(&xi.clone().transpose()).unwrap();

        let e = bce(
            yi.at(vec![0, 0]).unwrap().clone(),
            y_pred.at(vec![0, 0]).unwrap().clone(),
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
        e.clear_all_caches();
        e.backward(1.0);
        optim.update(itr);
    }

    for (xi, yi) in xor.iter() {
        let y_true = yi.at(vec![0, 0]).unwrap();
        println!(
            "{:?} -> true: {}",
            (
                xi.at(vec![0, 0]).unwrap().resolve(),
                xi.at(vec![1, 0]).unwrap().resolve()
            ),
            y_true.resolve()
        );
        let y_pred = model
            .forward(&xi.clone().transpose())
            .unwrap()
            .at(vec![0, 0])
            .unwrap()
            .clone();
        println!(
            "{:?} -> pred: {}",
            (
                xi.at(vec![0, 0]).unwrap().resolve(),
                xi.at(vec![1, 0]).unwrap().resolve()
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
