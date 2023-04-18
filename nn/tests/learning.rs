use nn::{serial,dense_layer::DenseLayer,relu_layer::ReluLayer,optim::OptimSGD, sigmoid_layer::SigmoidLayer};
use rand::{ thread_rng, Rng, seq::SliceRandom };

#[test]
fn model_inference() {
    let mut rng = thread_rng();
    let mut model = serial::Serial::new(vec![
        Box::new(DenseLayer::new(2, 5)),
        Box::new(ReluLayer::new()),
        Box::new(DenseLayer::new(5, 2)),
        Box::new(SigmoidLayer::new())
    ]);
    let mut x:Vec<f64> = Vec::new();
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
        Box::new(DenseLayer::new(5, 1)),
        Box::new(SigmoidLayer::new())
    ]);

    let max_itr = 200000;
    let optim = OptimSGD::new(0.01,max_itr);

    let xor = vec![
        ([0.0,0.0],[0.0]),
        ([0.1,0.0],[1.0]),
        ([0.0,1.0],[1.0]),
        ([1.0,1.0],[0.0])
    ];
    let mut e_log = String::new();
    for itr in 0..max_itr {
        
        // let sample = xor.choose(&mut rng);
        // let (xi,yi) = sample.unwrap();
        // let y_pred = model.forward(&xi.to_vec());
        // let e = mse(&yi.to_vec(), &y_pred);
        // e_log += &e.to_string(); e_log += "\n";
        // let e_grad = mse_prime(&yi.to_vec(), &y_pred);

        let (xi,yi) = xor[itr % 4];
        let y_pred = model.forward(&xi.to_vec());
        let e = bce(yi.to_vec()[0], y_pred[0]);
        e_log += &e.to_string(); e_log += "\n";
        let e_grad = vec![bce_prime(yi.to_vec()[0], y_pred[0])];

        model.zero_grad();
        model.backward(e_grad).unwrap();
        optim.update(&mut model, itr.try_into().unwrap());
        
    }
    // println!("{}",e_log);
    for (xi,yi) in xor.iter() {
        println!("{:?} -> true: {}",xi,yi.to_vec()[0]);
        println!("{:?} -> pred: {}",xi, model.forward(&xi.to_vec())[0]);
    }
}

fn bce(y: f64, y_pred: f64) -> f64 {
    -(y * (y_pred+0.0001).ln() + (1.0 - y) * (1.0 - (y_pred-0.0001)).ln())
}

fn bce_prime(y: f64, y_pred: f64) -> f64 {
    -((y/(y_pred+0.0001)) + (y - 1.0)/(1.0 - (y_pred-0.0001)))
}

fn binary_x_entropy(y: &Vec<Vec<f64>>, y_pred: &Vec<Vec<f64>>) -> Vec<f64> {
    let batch_size = y.len();
    let y_len = y[0].len();
    let mut mean_loss = vec![0.0; y[0].len()];
    for s in 0..batch_size {
        for i in 0..y_len {
            mean_loss[i] += (y[s][i] * y_pred[s][i].ln() + (1.0 - y[s][i]) * (1.0 - y_pred[s][i]).ln())
        }
    }
    for el in &mut mean_loss {
        *el /= -1.0 * batch_size as f64;
    }
    mean_loss
}

fn binary_x_entropy_prime(y: &Vec<Vec<f64>>, y_pred: &Vec<Vec<f64>>) -> Vec<f64> {
    let batch_size = y.len();
    let y_len = y[0].len();
    let mut mean_loss_prime = vec![0.0; y[0].len()];
    for s in 0..batch_size {
        for i in 0..y_len {
            mean_loss_prime[i] += y_pred[s][i] - y[s][i];
        }
        // grad.push(-((y[i]/y_pred[i]) + (y[i] - 1.0)/(1.0 - y_pred[i])));
    }
    for el in &mut mean_loss_prime {
        *el /= batch_size as f64;
    }
    mean_loss_prime
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