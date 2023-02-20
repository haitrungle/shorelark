use fastrand::Rng;

pub(crate) struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    pub(crate) fn propagate(&self, inputs: &[f32]) -> f32 {
        // TODO: handle this better
        assert_eq!(inputs.len(), self.weights.len());

        let output = self.bias
            + inputs
                .iter()
                .zip(&self.weights)
                .map(|(input, weight)| input * weight)
                .sum::<f32>();

        output.max(0.0)
    }

    pub(crate) fn random(rng: &Rng, output_size: usize) -> Self {
        // in the range [-1, 1]
        let bias = rng.f32() * 2.0 - 1.0;

        let weights = (0..output_size).map(|_| rng.f32() * 2.0 - 1.0).collect();

        Self { bias, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod random {
        use super::*;

        #[test]
        fn test() {
            let rng = Rng::with_seed(4);
            let neuron = Neuron::random(&rng, 4);

            approx::assert_relative_eq!(neuron.bias, 0.5594833);

            approx::assert_relative_eq!(
                neuron.weights.as_slice(),
                [-0.848562, 0.9985919, -0.6743002, 0.8647733].as_ref()
            );
        }
    }

    mod propagate {
        use super::*;

        #[test]
        fn test() {
            let neuron = Neuron {
                bias: 0.5,
                weights: vec![-0.3, 0.8],
            };

            // Ensures `.max()` (our ReLU) works:
            approx::assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0);

            // `0.5` and `1.0` chosen by a fair dice roll:
            approx::assert_relative_eq!(
                neuron.propagate(&[0.5, 1.0]),
                (-0.3 * 0.5) + (0.8 * 1.0) + 0.5
            );
        }
    }
}
