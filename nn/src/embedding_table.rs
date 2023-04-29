use rand::{thread_rng, distributions::Distribution };
use statrs::distribution::Normal;
pub struct EmbeddingTable {
    table:Vec<Row>
}

struct Row {
    items:Vec<f64>
}

impl EmbeddingTable {
    pub fn new(shape: (usize,usize)) -> EmbeddingTable {
        let mut table = Vec::new();
        for _ in 0..shape.0 {
            table.push(Row::new(shape.1));
        }
        EmbeddingTable { table }
    }

    pub fn get_emb(&self, x:&Vec<Vec<usize>>) -> Vec<Vec<Vec<f64>>> {
        let mut emb = Vec::new();
        for sample in x.iter() {
            let mut out_sample = Vec::new();
            for item in sample.iter() {
                out_sample.push(self.table[*item].to_vec());
            }
            emb.push(out_sample);
        }
        emb
    }
}

impl Row {
    fn new(len:usize) -> Row {
        let mut rng = thread_rng();
        let norm = Normal::new(0.0,1.0).unwrap();
        let mut items = Vec::new();
        for _ in 0..len {
            items.push(norm.sample(&mut rng))
        }
        Row { items }
    }
    fn to_vec(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for item in self.items.iter() {
            out.push(*item);
        }
        out
    }
}