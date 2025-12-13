pub trait Drafter {
    fn update(&mut self, context: &[u32]);
    fn draft(&mut self) -> (Vec<u32>, Vec<u32>);
}

pub struct Empty {}

impl Drafter for Empty {
    fn update(&mut self, _context: &[u32]) {}

    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
        (vec![], vec![])
    }
}
