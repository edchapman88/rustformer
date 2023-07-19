use micrograd::{Node, Value};
use nn::{
    dense_layer::DenseLayer, optim::OptimSGD, relu_layer::ReluLayer, serial,
    sigmoid_layer::SigmoidLayer,
};
use rand::{seq::SliceRandom, thread_rng, Rng};

#[test]
fn model_inference() {
    let mut rng = thread_rng();
    let mut model = serial::Serial::new(vec![
        Box::new(DenseLayer::new(2, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(5, 2)),
        Box::new(SigmoidLayer::new()),
    ]);
    let mut x: Vec<f64> = Vec::new();
    for _ in 0..4 {
        x.push(rng.gen())
    }
    let y = model.forward(&x);
    // println!("{:?}",y);
}

#[test]
fn model_learning() {
    let mut model = serial::Serial::new(vec![
        Box::new(DenseLayer::new(2, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(5, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(5, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(5, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(5, 1)),
        Box::new(SigmoidLayer::new()),
    ]);

    let max_itr = 200000;
    let optim = OptimSGD::new(0.01, max_itr);

    let xor = vec![
        ([0.0, 0.0], [0.0]),
        ([0.1, 0.0], [1.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];
    let mut e_log = String::new();
    for itr in 0..max_itr {
        // let sample = xor.choose(&mut rng);
        // let (xi,yi) = sample.unwrap();
        // let y_pred = model.forward(&xi.to_vec());
        // let e = mse(&yi.to_vec(), &y_pred);
        // e_log += &e.to_string(); e_log += "\n";
        // let e_grad = mse_prime(&yi.to_vec(), &y_pred);

        let (xi, yi) = xor[itr % 4];
        let y_pred = model.forward(&xi.to_vec());
        // final dense layer of model has output dimension of 1
        // so sigmoid input and output dimension is 1
        // println!("{:?}", y_pred);
        let mut e = bce(yi.to_vec()[0], y_pred[0]);
        e_log += &e.resolve().to_string();
        e_log += "\n";

        let op_leaves = e.backward(1.0);
        assert_eq!(op_leaves[2].data, y_pred[0]);
        assert_eq!(op_leaves[7].data, y_pred[0]);
        let e_grad = vec![op_leaves[2].grad + op_leaves[7].grad];

        model.zero_grad();
        model.backward(e_grad).unwrap();
        optim.update(&mut model, itr);
    }
    // println!("{}", e_log);
    for (xi, yi) in xor.iter() {
        println!("{:?} -> true: {}", xi, yi.to_vec()[0]);
        println!("{:?} -> pred: {}", xi, model.forward(&xi.to_vec())[0]);
    }
}

fn bce(y_float: f64, y_pred_float: f64) -> Node {
    // -1 * [ y * (y_pred + 0.0001).ln()    +    (1 - y) * (1 - (y_pred - 0.0001)).ln() ]
    let y_pred = Value::new(y_pred_float);
    let y = Value::new(y_float);
    &Value::new(-1.0)
        * (&y * (&y_pred + &Value::new(0.0001)).ln()
            + (&Value::new(1.0) - &y) * (&Value::new(1.0) - (&y_pred - &Value::new(0.0001))).ln())
}

fn binary_x_entropy(y: &Vec<Vec<f64>>, y_pred: &Vec<Vec<f64>>) -> Vec<f64> {
    // TODO: refactor to be auto-differentiable (see bce)
    let batch_size = y.len();
    let y_len = y[0].len();
    let mut mean_loss = vec![0.0; y[0].len()];
    for s in 0..batch_size {
        for i in 0..y_len {
            mean_loss[i] +=
                (y[s][i] * y_pred[s][i].ln() + (1.0 - y[s][i]) * (1.0 - y_pred[s][i]).ln())
        }
    }
    for el in &mut mean_loss {
        *el /= -1.0 * batch_size as f64;
    }
    mean_loss
}
