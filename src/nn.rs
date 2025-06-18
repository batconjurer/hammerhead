//! A DCNN for training Hnefatafl using self-play.
//!
//! The data for an ongoing Hnefatafl game is as follows:
//!  * An 11 x 11 board with attacker positions
//!  * An 11 x 11 board with defender positions
//!  * A count of the number of times this position has been visited
//!  * A boolean indication if it is the attacker's turn
//!  * A total move count
//!
//! Following the approach of AlphaZero, c.f. https://arxiv.org/pdf/1712.01815,
//! we represent board state as an (2T + 3) x 11 x 11 image stack, i.e.
//! 2T + 3 input channels of 11 x 11 boards where T is the amount of historical
//! data, currently only a value of 1 is supported.
//!
//! Two of the 11 x 11 slices are for the piece positions. The last 3 slice
//! contains the game metadata.

use std::default;
use std::env::var;
use std::path::{Path, PathBuf};
use candle_core::{DType, Device, Module, Tensor, Shape};
use candle_nn::ops::{dropout, log_softmax};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, VarBuilder, batch_norm, conv2d_no_bias, linear_no_bias, Optimizer, VarMap, Init};

/// A trainable DCNN for Hnefatafl
pub struct TaflNNet {
    convolutions: [NormedConv2d; 4],
    linear_layers: [NormedLinear; 3],
    optimizer: candle_nn::AdamW,
    backend: PersistentVarMap,
}

impl TaflNNet {
    /// Initialize the DCNN architecture
    pub fn new(backend: PersistentVarMap) -> Self {
        // the convolution layers
        let mut convolutions = [
            NormedConv2d::new(5, 64, 1, &backend),
            NormedConv2d::new( 64, 128, 1, &backend),
            NormedConv2d::new(  128, 256, 0, &backend),
            NormedConv2d::new(  256, 512, 0, &backend),
        ];
        // the linear layers
        let linear_layers = [
            NormedLinear::new(512, 1024, true, &backend),
            NormedLinear::new( 1024, 2 * 11usize.pow(4), true, &backend),
            NormedLinear::new( 2 * 11usize.pow(4), 1, false, &backend),
        ];
        let optimizer = candle_nn::AdamW::new(
            backend.inner.all_vars(),
            candle_nn::ParamsAdamW {
                lr: 1e-2,
                ..Default::default()
            }
        ).unwrap();
        Self {
            convolutions,
            linear_layers,
            optimizer,
            backend,
        }
    }

    /// Train the model with input compared against target for
    /// the given number of epochs.
    ///
    /// The input is the above described Hnefatafl image stack.
    /// The output is an evaluation of the position for the
    /// current player, represented as a probability computed
    /// via an MCTS.
    pub fn train(
        &self,
        input: &Tensor,
        target: &Tensor,
        optimizer: &mut candle_nn::AdamW,
        epochs: usize,
    ) -> candle_core::Result<()> {
        for _ in 0..epochs {
            let output = self.forward(input)?;
            let loss = candle_nn::loss::mse(&output.squeeze(1)?, target)?;
            optimizer.backward_step(&loss)?;
        }
        Ok(())
    }
}

impl Module for TaflNNet {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut xs = xs.reshape((5, 11, 11))?;
        for conv in &self.convolutions {
            xs = conv.forward(&xs)?;
        }
        for ll in  &self.linear_layers {
            xs = ll.forward(&xs)?;
        }
        xs.tanh()
    }
}

/// A convolution layer that is normed during forwarding
pub struct NormedConv2d {
    conv: Conv2d,
    norm: BatchNorm,
}

impl NormedConv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        padding: usize,
        backend: &PersistentVarMap,
    ) -> Self {
        Self {
            conv: conv2d_no_bias(
                in_channels,
                out_channels,
                3,
                Conv2dConfig {
                    padding,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
                VarBuilder::from_varmap(&backend.inner, DType::U8, &Device::Cpu)
            )
            .unwrap(),
            norm:  batch_norm(
                64,
                BatchNormConfig {
                    eps: 0.00001,
                    remove_mean: false,
                    affine: true,
                    momentum: 0.1,
                },
                VarBuilder::from_varmap(&backend.inner, DType::U8, &Device::Cpu),
            ).unwrap()
        }
    }
}

impl Module for NormedConv2d {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.norm.forward_train(&self.conv.forward(&xs)?)?;
        xs.relu()
    }
}

/// A linear layer that is normed during forwarding
pub struct NormedLinear {
    layer: Linear,
    norm: BatchNorm,
    dropout: bool,
}

impl NormedLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        dropout: bool,
        backend: &PersistentVarMap,
    ) -> Self {
        Self {
            layer: linear_no_bias(
                in_dim,
                out_dim,
                VarBuilder::from_varmap(&backend.inner, DType::U8, &Device::Cpu),
            )
            .unwrap(),
            norm:  batch_norm(
                out_dim,
                BatchNormConfig {
                    eps: 0.00001,
                    remove_mean: false,
                    affine: true,
                    momentum: 0.1,
                },
                VarBuilder::from_varmap(&backend.inner, DType::U8, &Device::Cpu),
            ).unwrap(),
            dropout
        }
    }
}

impl Module for NormedLinear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut xs = self.norm.forward_train(&self.layer.forward(&xs)?)?;
        xs = xs.relu()?;
        if self.dropout {
            xs = dropout(&xs, 0.2)?;
        }
        Ok(xs)
    }
}

/// A map of tensor data backed by a file
pub struct PersistentVarMap {
    inner: VarMap,
    path: PathBuf,
}

impl PersistentVarMap {
    pub fn load_or_new(path: impl AsRef<Path>) -> Self {
        let mut varmap = VarMap::new();
        _ = varmap.load(path.as_ref());
        Self{inner: varmap, path: path.as_ref().to_path_buf() }
    }

    pub fn save(&self) -> candle_core::Result<()> {
        self.inner.save(&self.path)
    }
}

impl SimpleBackend for PersistentVarMap {
    fn get(&self, s: Shape, name: &str, h: Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        <VarMap as SimpleBackend>::get(&self.inner, s, name, h, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_tensor(name)
    }
}

impl Drop for PersistentVarMap {
    fn drop(&mut self) {
        _ = self.save();
    }
}