mod layer;
mod neuron;

use fastrand::Rng;

use crate::layer::*;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers.iter().fold(inputs, |acc, x| x.propagate(acc))
    }

    pub fn random(rng: &Rng, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0].neurons, layers[1].neurons))
            .collect();

        Self { layers }
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
            let layer_topologies = [
                LayerTopology { neurons: 3 },
                LayerTopology { neurons: 4 },
                LayerTopology { neurons: 2 },
            ];
            let network = Network::random(&rng, &layer_topologies);
            let inputs = vec![0.3, 0.6, 0.8];

            approx::assert_relative_eq!(
                network.propagate(inputs).as_slice(),
                &[0.24046704, 0.11406484].as_ref()
            );
        }
    }
}
