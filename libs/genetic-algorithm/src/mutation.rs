use fastrand::Rng;

use crate::chromosome::Chromosome;

pub trait MutationMethod {
    fn mutate(&self, rng: &Rng, child: &mut Chromosome);
}

#[derive(Clone, Debug)]
pub struct GaussianMutation {
    /// Probability of changing a gene:
    /// - 0.0 = no genes will be touched
    /// - 1.0 = all genes will be touched
    chance: f32,

    /// Magnitude of that change:
    /// - 0.0 = touched genes will not be modified
    /// - 3.0 = touched genes will be += or -= by at most 3.0
    coeff: f32,
}

impl GaussianMutation {
    pub fn new(chance: f32, coeff: f32) -> Self {
        assert!(chance >= 0.0 && chance <= 1.0);

        Self { chance, coeff }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &Rng, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            let sign = if rng.bool() { -1.0 } else { 1.0 };

            if rng.f32() <= self.chance {
                *gene += sign * self.coeff * rng.f32();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn actual(chance: f32, coeff: f32) -> Vec<f32> {
        let mut child = vec![1.0, 2.0, 3.0, 4.0, 5.0].into_iter().collect();

        let mut rng = Rng::with_seed(15);

        GaussianMutation::new(chance, coeff).mutate(&mut rng, &mut child);

        child.into_iter().collect()
    }

    mod given_zero_chance {
        fn actual(coeff: f32) -> Vec<f32> {
            super::actual(0.0, coeff)
        }
        mod and_zero_coefficient {
            use super::*;
            #[test]
            fn does_not_change_the_original_chromosome() {
                let actual = actual(0.0);
                let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
            }
        }

        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn does_not_change_the_original_chromosome() {
                let actual = actual(0.5);
                let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
            }
        }
    }

    mod given_fifty_fifty_chance {
        fn actual(coeff: f32) -> Vec<f32> {
            super::actual(0.5, coeff)
        }
        mod and_zero_coefficient {
            use super::*;
            #[test]
            fn does_not_change_the_original_chromosome() {
                let actual = actual(0.0);
                let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
            }
        }

        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn slightly_changes_the_original_chromosome() {
                let actual = actual(1.0);
                let expected = vec![1.0, 2.9207408, 3.6450171, 4.0, 5.0];

                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
            }
        }
    }

    mod given_max_chance {
        fn actual(coeff: f32) -> Vec<f32> {
            super::actual(1.0, coeff)
        }

        mod and_zero_coefficient {
            use super::*;

            #[test]
            fn does_not_change_the_original_chromosome() {
                let actual = actual(0.0);
                let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
            }
        }

        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn entirely_changes_the_original_chromosome() {
                let actual = actual(1.0);
                let expected = vec![1.4472078, 1.1634266, 3.8299372, 4.9581404, 5.4619694];

                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
            }
        }
    }
}
