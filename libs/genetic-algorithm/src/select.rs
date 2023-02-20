use fastrand::Rng;

use crate::individual::Individual;

pub trait SelectionMethod {
    fn select<'a, I>(&self, rng: &Rng, population: &'a [I]) -> &'a I
    where
        I: Individual;
}

pub struct RouletteWheelSelection;

impl RouletteWheelSelection {
    pub fn new() -> Self {
        Self
    }
}

impl SelectionMethod for RouletteWheelSelection {
    fn select<'a, I>(&self, rng: &Rng, population: &'a [I]) -> &'a I
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        let cumulative_fitness: Vec<f32> = population
            .iter()
            .scan(0.0, |acc, x| {
                *acc = *acc + x.fitness();
                Some(*acc)
            })
            .collect();
        let total_fitness = cumulative_fitness.last().unwrap();

        let choice = rng.f32() * total_fitness;

        let index_choice = cumulative_fitness.partition_point(|&x| x < choice);

        &population[index_choice]
    }
}

#[cfg(test)]
mod tests {
    use crate::individual::TestIndividual;
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn test() {
        let rng = Rng::with_seed(19);
        let population = vec![
            TestIndividual::new(2.0),
            TestIndividual::new(1.0),
            TestIndividual::new(4.0),
            TestIndividual::new(3.0),
        ];

        let method = RouletteWheelSelection::new();

        let actual_histogram: BTreeMap<i32, _> = (0..1000)
            .map(|_| method.select(&rng, &population))
            .fold(Default::default(), |mut histogram, individual| {
                *histogram.entry(individual.fitness() as _).or_default() += 1;
                histogram
            });

        let expected_histogram = BTreeMap::from_iter(vec![
            // (fitness, how many times this fitness has been chosen)
            (1, 100),
            (2, 202),
            (3, 280),
            (4, 418),
        ]);

        assert_eq!(actual_histogram, expected_histogram);
    }
}
