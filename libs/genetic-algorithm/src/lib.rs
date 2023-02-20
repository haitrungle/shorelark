mod chromosome;
mod crossover;
mod individual;
mod mutation;
mod select;

use fastrand::Rng;

use crossover::CrossoverMethod;
use individual::Individual;
use mutation::MutationMethod;
use select::SelectionMethod;

pub struct GeneticAlgorithm<S> {
    selection_method: S,
    crossover_method: Box<dyn CrossoverMethod>,
    mutation_method: Box<dyn MutationMethod>,
}

impl<S> GeneticAlgorithm<S>
where
    S: SelectionMethod,
{
    pub fn new(
        selection_method: S,
        crossover_method: impl CrossoverMethod + 'static,
        mutation_method: impl MutationMethod + 'static,
    ) -> Self {
        Self {
            selection_method,
            crossover_method: Box::new(crossover_method),
            mutation_method: Box::new(mutation_method),
        }
    }

    pub fn evolve<I>(&self, rng: &Rng, population: &[I]) -> Vec<I>
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        (0..population.len())
            .map(|_| {
                let parent_a = self.selection_method.select(rng, population).chromosome();

                let parent_b = self.selection_method.select(rng, population).chromosome();

                let mut offspring = self.crossover_method.crossover(rng, parent_a, parent_b);

                self.mutation_method.mutate(rng, &mut offspring);

                I::create(offspring)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        crossover::UniformCrossover, individual::TestIndividual, mutation::GaussianMutation,
        select::RouletteWheelSelection,
    };

    use super::*;

    #[test]
    fn test() {
        let rng = Rng::with_seed(19);

        let ga = GeneticAlgorithm::new(
            RouletteWheelSelection::new(),
            UniformCrossover::new(),
            GaussianMutation::new(0.5, 0.5),
        );

        let mut population = vec![
            individual(&[0.0, 0.0, 0.0]), // fitness = 0.0
            individual(&[1.0, 1.0, 1.0]), // fitness = 3.0
            individual(&[1.0, 2.0, 1.0]), // fitness = 4.0
            individual(&[1.0, 2.0, 4.0]), // fitness = 7.0
        ];

        for _ in 0..10 {
            population = ga.evolve(&rng, &population);
        }

        let expected_population = vec![
            individual(&[0.91706, 2.1020525, 5.088519]),
            individual(&[1.590559, 2.1020525, 4.5034056]),
            individual(&[0.7727975, 1.8319678, 5.133454]),
            individual(&[0.5474399, 1.8319678, 4.860874]),
        ];

        assert_eq!(population, expected_population);
    }

    fn individual(genes: &[f32]) -> TestIndividual {
        let chromosome = genes.iter().cloned().collect();

        TestIndividual::create(chromosome)
    }
}
