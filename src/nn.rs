//! A DCNN for training Hnefatafl using self-play.
//!
//! The data for an ongoing Hnefatafl game is as follows:
//!  * An 11 x 11 board with attacker positions
//!  * An 11 x 11 board with defender positions
//!  * A boolean indication if it is the attacker's turn
//!  * A total move count
//!
//! Following the approach of AlphaZero, c.f. https://arxiv.org/pdf/1712.01815,
//! we represent board state as an (2T + 3) x 11 x 11 image stack, i.e.
//! 2T + 2 input channels of 11 x 11 boards where T is the amount of historical
//! data, currently only a value of 1 is supported.
//!
//! Two of the 11 x 11 slices are for the piece positions. The last 3 slice
//! contains the game metadata.
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::ops::dropout;

use candle_nn::{BatchNorm, Conv2d, Conv2dConfig, Linear, Optimizer, VarBuilder, VarMap};

/// A trainable DCNN for Hnefatafl
pub struct TaflNNet {
    convolutions: [NormedConv2d; 4],
    linear_layers: [NormedLinear; 4],
    optimizer: candle_nn::AdamW,
    #[allow(dead_code)]
    backend: PersistentVarMap,
}

impl TaflNNet {
    /// Initialize the DCNN architecture
    pub fn new(model_files: impl AsRef<Path>) -> Self {
        let backend = PersistentVarMap::load_or_new(model_files);
        // the convolution layers
        let convolutions = [
            NormedConv2d::new(4, 64, 1, &backend),
            NormedConv2d::new(64, 128, 1, &backend),
            NormedConv2d::new(128, 256, 0, &backend),
            NormedConv2d::new(256, 512, 0, &backend),
        ];
        // the linear layers
        let linear_layers = [
            NormedLinear::new(512, 1024, true, &backend),
            NormedLinear::new(1024, 2 * 11usize.pow(4), true, &backend),
            NormedLinear::new(2 * 11usize.pow(4), 1, false, &backend),
            NormedLinear::new(7 * 7, 1, false, &backend),
        ];
        let optimizer = candle_nn::AdamW::new(
            backend.inner.all_vars(),
            candle_nn::ParamsAdamW {
                lr: 1e-2,
                ..Default::default()
            },
        )
        .unwrap();
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
        &mut self,
        input: &Tensor,
        target: &Tensor,
        epochs: usize,
    ) -> candle_core::Result<()> {
        for ep in 0..epochs {
            let output = self
                .forward(input)
                .inspect_err(|e| println!("Could not train on input: {e}"))?;
            let loss = candle_nn::loss::mse(&output, target)
                .inspect_err(|e| println!("Could not compute loss: {e}"))?;
            if ep.rem_euclid(10) == 0 {
                let o = output.max(0).unwrap().to_scalar::<f64>().unwrap();
                let t = target.max(0).unwrap().to_scalar::<f64>().unwrap();
                let l = loss.to_scalar::<f64>().unwrap();
                println!("Output: {o}, target: {t}, loss: {l}")
            }
            self.optimizer
                .backward_step(&loss)
                .inspect_err(|e| println!("Could not run optimizer: {e}"))?;
        }
        Ok(())
    }
}

impl Module for TaflNNet {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut xs = xs.reshape(((), 4, 11, 11))?;
        //let mut xs = xs.clone();
        for conv in &self.convolutions {
            xs = conv.forward(&xs)?;
        }
        xs = xs.reshape((49, 512))?;
        for (layer, ll) in self.linear_layers.iter().enumerate() {
            xs = ll.forward(&xs)?;
            if layer == 2 {
                xs = xs.reshape((1, 49))?;
            }
        }
        xs = xs.reshape(1)?;
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
        let vb = VarBuilder::from_varmap(&backend.inner, DType::F64, &Device::Cpu);
        Self {
            conv: {
                let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
                let cfg = Conv2dConfig {
                    padding,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                };
                let ws = vb
                    .get_with_hints(
                        (out_channels, in_channels / cfg.groups, 3, 3),
                        // we need to make sure that variable names don't conflict
                        &format!("conv2d_weight_{in_channels}_{out_channels}"),
                        init_ws,
                    )
                    .unwrap();
                Conv2d::new(ws, None, cfg)
            },
            norm: {
                use candle_nn::Init;

                let running_mean = vb
                    .get_with_hints(
                        out_channels,
                        &format!("running_mean_convnorm_{out_channels}"),
                        Init::Const(0.),
                    )
                    .unwrap();
                let running_var = vb
                    .get_with_hints(
                        out_channels,
                        &format!("running_var_convnorm_{out_channels}"),
                        Init::Const(1.),
                    )
                    .unwrap();

                let weight = vb
                    .get_with_hints(
                        out_channels,
                        &format!("weight_convnorm_{out_channels}"),
                        Init::Const(1.),
                    )
                    .unwrap();
                let bias = vb
                    .get_with_hints(
                        out_channels,
                        &format!("bias_convnorm_{out_channels}"),
                        Init::Const(0.),
                    )
                    .unwrap();

                BatchNorm::new_with_momentum(
                    out_channels,
                    running_mean,
                    running_var,
                    weight,
                    bias,
                    0.00001,
                    0.1,
                )
                .unwrap()
            },
        }
    }
}

impl Module for NormedConv2d {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = self.norm.forward_train(&xs)?;
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
    pub fn new(in_dim: usize, out_dim: usize, dropout: bool, backend: &PersistentVarMap) -> Self {
        let vb = VarBuilder::from_varmap(&backend.inner, DType::F64, &Device::Cpu);
        Self {
            layer: {
                let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
                let ws = vb
                    .get_with_hints(
                        (out_dim, in_dim),
                        &format!("weight_linear_{in_dim}_{out_dim}"),
                        init_ws,
                    )
                    .unwrap();
                Linear::new(ws, None)
            },
            norm: {
                use candle_nn::Init;

                let running_mean = vb
                    .get_with_hints(
                        out_dim,
                        &format!("running_mean_linnorm_{out_dim}"),
                        Init::Const(0.),
                    )
                    .unwrap();
                let running_var = vb
                    .get_with_hints(
                        out_dim,
                        &format!("running_var_linnorm_{out_dim}"),
                        Init::Const(1.),
                    )
                    .unwrap();

                let weight = vb
                    .get_with_hints(
                        out_dim,
                        &format!("weight_linnorm_{out_dim}"),
                        Init::Const(1.),
                    )
                    .unwrap();
                let bias = vb
                    .get_with_hints(out_dim, &format!("bias_linnorm_{out_dim}"), Init::Const(0.))
                    .unwrap();

                BatchNorm::new_with_momentum(
                    out_dim,
                    running_mean,
                    running_var,
                    weight,
                    bias,
                    0.00001,
                    0.1,
                )
                .unwrap()
            },
            dropout,
        }
    }
}

impl Module for NormedLinear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.layer.forward(xs)?;
        let mut xs = self.norm.forward_train(&xs)?;
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
        Self {
            inner: varmap,
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn save(&self) -> candle_core::Result<()> {
        self.inner.save(&self.path)
    }
}

impl Drop for PersistentVarMap {
    fn drop(&mut self) {
        _ = self.save();
    }
}
