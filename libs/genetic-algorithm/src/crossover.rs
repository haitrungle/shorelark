use fastrand::Rng;

use crate::chromosome::Chromosome;

pub trait CrossoverMethod {
    fn crossover(&self, rng: &Rng, parent_a: &Chromosome, parent_b: &Chromosome) -> Chromosome;
}

#[derive(Clone, Debug)]
pub struct UniformCrossover;

impl UniformCrossover {
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverMethod for UniformCrossover {
    fn crossover(&self, rng: &Rng, parent_a: &Chromosome, parent_b: &Chromosome) -> Chromosome {
        assert_eq!(parent_a.len(), parent_b.len());

        parent_a
            .iter()
            .zip(parent_b.iter())
            .map(|(&a, &b)| if rng.bool() { a } else { b })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let rng = Rng::with_seed(41);
        let parent_a: Chromosome = (1..=100).map(|n| n as f32).collect();
        let parent_b: Chromosome = (1..=100).map(|n| -n as f32).collect();

        let child = UniformCrossover::new().crossover(&rng, &parent_a, &parent_b);

        // Number of genes different between `child` and `parent_a`
        let diff_a = child.iter().zip(parent_a).filter(|(c, p)| *c != p).count();

        // Number of genes different between `child` and `parent_b`
        let diff_b = child.iter().zip(parent_b).filter(|(c, p)| *c != p).count();

        assert_eq!(diff_a, 51);
        assert_eq!(diff_b, 49);
    }
}
