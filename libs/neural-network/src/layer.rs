use fastrand::Rng;

use crate::neuron::*;

pub struct LayerTopology {
    pub neurons: usize,
}

pub(crate) struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub(crate) fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons.iter().map(|n| n.propagate(&inputs)).collect()
    }

    pub(crate) fn random(rng: &Rng, input_size: usize, output_size: usize) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::random(rng, input_size))
            .collect();

        Self { neurons }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod propagate {
        use super::*;

        #[test]
        fn test() {
            let rng = Rng::with_seed(4);
            let layer = Layer::random(&rng, 3, 4);
            let inputs = vec![0.3, 0.6, 0.8];

            approx::assert_relative_eq!(
                layer.propagate(inputs).as_slice(),
                &[0.36462972, 0.44992748, 0.15580672, 0.0].as_ref()
            );
        }
    }
}
