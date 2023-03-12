use nn::{serial,dense_layer::DenseLayer,relu_layer::ReluLayer,optim::OptimSGD};
use rand::{ thread_rng, Rng, seq::SliceRandom };

#[test]
fn model_inference() {
    let mut rng = thread_rng();
    let mut model = serial::Serial::new(vec![
        Box::new(DenseLayer::new(2, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(5, 2))
    ]);
    let mut x:Vec<f64> = Vec::new();
    for _ in 0..4 {
       x.push(rng.gen())
    }
    let y = model.forward(&x);
    println!("{:?}",y);
}

#[test]
fn model_learning() {
    let mut rng = thread_rng();
    let mut model = serial::Serial::new(vec![
        Box::new(DenseLayer::new(2, 3)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(3, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(5, 1))
    ]);

    let optim = OptimSGD::new(0.00001);

    let xor = vec![
        ([0.0,0.0],[1.0]),
        ([0.1,0.0],[0.0]),
        ([0.0,1.0],[0.0]),
        ([1.0,1.0],[1.0])
    ];
    let mut e_log = String::new();
    for _ in 0..500 {
        let sample = xor.choose(&mut rng);
        let (x,y) = sample.unwrap();
        let y_pred = model.forward(&x.to_vec());
        let e = mse(&y.to_vec(), &y_pred);
        e_log += &e.to_string(); e_log += "\n";

        let e_grad = mse_prime(&y.to_vec(), &y_pred);
        model.backward(e_grad).unwrap();
        optim.update(&mut model);
        model.zero_grad();
    }
    println!("{}",e_log);
}

fn mse(y: &Vec<f64>, y_pred: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..y.len() {
        sum += (y[i] - y_pred[i]).powi(2);
    }
    sum / y.len() as f64
}

fn mse_prime(y: &Vec<f64>, y_pred: &Vec<f64>) -> Vec<f64> {
    let mut grad = Vec::new();
    for i in 0..y.len() {
        grad.push((y_pred[i] - y[i]) * 2.0 / y.len() as f64);
    }
    grad
}