| Hyperparameter | Description | REINFORCE | SODA | DLMI($\tau=100$)-TF | DLMI($\tau=100$) |
|---|---|--|-|-|-|
| baseline_backend | Model backend for SODA/GCG: 'hf' (HuggingFace) or 'tl' (transformer_lens) | [] | [hf] | [] | [] |
| batch_size | Batch size for optimization | [8] | [8] | [8] | [8] |
| bptt | Whether to use backpropagation through time | [false] | [] | [false] | [false] |
| bptt_hidden_state_conditioning | Whether to condition BPTT on hidden states | [false] | [] | [false] | [false] |
| bptt_learnable_temperature | Whether to learn the BPTT temperature parameter | [false] | [] | [false] | [false] |
| bptt_stgs_hard | Whether to use hard ST-GS for BPTT | [false] | [] | [false] | [false] |
| bptt_temperature | Temperature for BPTT Gumbel-Softmax | [100] | [] | [100] | [100] |
| dataset_path | Path to the dataset file containing target sentences | [./data/smollm3-3b-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1] | [./data/smollm3-3b-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1] | [data/smollm3-3b-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1] | [data/smollm3-3b-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1] |
| decouple_learnable_temperature | Whether to learn multiple decouple temperature parameters, one for each learnable input. | [false] | [] | [true] | [true] |
| epochs | Number of optimization epochs | [2048] | [2048] | [2048] | [2048] |
| gradient_checkpointing | Whether to use gradient checkpointing | [False] | [False] | [False] | [False] |
| gradient_estimator | Gradient estimator to use for optimizing the discrete inputs | [reinforce] | [] | [stgs] | [stgs] |
| learnable_temperature | Whether to learn the temperature parameter | [false] | [] | [true] | [true] |
| learning_rate | Learning rate for optimization | [0.1] | [0.03] | [0.1] | [0.1] |
| losses | Loss function(s) to use for optimization | [crossentropy] | [crossentropy] | [crossentropy] | [crossentropy] |
| method | Optimization method to use | [] | [soda] | [] | [] |
| model_name | Name of the language model to use | [HuggingFaceTB/SmolLM3-3B-Base] | [HuggingFaceTB/SmolLM3-3B-Base] | [HuggingFaceTB/SmolLM3-3B-Base] | [HuggingFaceTB/SmolLM3-3B-Base] |
| model_precision | Precision of the language model to use | [full] | [full] | [full] | [full] |
| num_workers | Number of parallel workers (1 = sequential) | [1] | [1] | [1] | [1] |
| output_dir | Directory to save optimization results | [results/smollm3_2048_grid_REINFORCE_sweep] | [results/smollm3_3b_2048ep_SODA_grid_sweep] | [results/smollm3_3b_2048ep_grid_sweep] | [results/smollm3_3b_2048ep_grid+tf_sweep] |
| plot_every | Frequency of plotting loss curves | [100000] | [100000] | [100000] | [100000] |
| reinforce_baseline_beta | Exponential moving average coefficient for the REINFORCE baseline | [0.9] | [] | [] | [] |
| reinforce_grad_variance_period | Epoch interval between two REINFORCE gradient variance measurements | [25] | [] | [] | [] |
| reinforce_grad_variance_samples | Number of REINFORCE gradient samples to use when estimating variance (>=2 enables metric) | [0] | [] | [] | [] |
| reinforce_reward_scale | Scaling factor applied to the REINFORCE policy loss | [1] | [] | [] | [] |
| reinforce_use_baseline | Enable running baseline to reduce REINFORCE variance | [true] | [] | [] | [] |
| seed | Random seed for reproducibility | [10, 20] | [10, 20, 30, 40] | [10, 20, 30, 40] | [10, 20, 30] |
| seq_len | Length of the prompt sequence to optimize | [80] | [80] | [80] | [80] |
| soda_beta1 | SODA: Adam beta1 parameter | [] | [0.9] | [] | [] |
| soda_beta2 | SODA: Adam beta2 parameter | [] | [0.995] | [] | [] |
| soda_bias_correction | SODA: use standard Adam bias correction | [] | [false] | [] | [] |
| soda_decay_rate | SODA: embedding decay rate per epoch | [] | [0.98] | [] | [] |
| soda_init_strategy | SODA: embedding initialization strategy | [] | [zeros] | [] | [] |
| soda_reinit_epoch | SODA: embedding reinitialization frequency | [] | [1500] | [] | [] |
| soda_reset_epoch | SODA: optimizer state reset frequency | [] | [50] | [] | [] |
| stgs_grad_bias_period | Epoch interval between two STGS bias measurements | [0] | [] | [0] | [0] |
| stgs_grad_bias_reference_baseline_beta | EMA beta for the reference REINFORCE baseline during bias estimation | [0.9] | [] | [0.9] | [0.9] |
| stgs_grad_bias_reference_reward_scale | Reward scale to apply in the reference REINFORCE bias estimator | [1] | [] | [1] | [1] |
| stgs_grad_bias_reference_samples | Number of REINFORCE reference samples per bias estimate (>=1 enables metric) | [0] | [] | [0] | [0] |
| stgs_grad_bias_reference_use_baseline | Enable REINFORCE baseline when computing STGS bias | [true] | [] | [true] | [true] |
| stgs_grad_bias_samples | Number of STGS gradient samples to average when estimating bias (>=1 enables metric) | [0] | [] | [0] | [0] |
| stgs_grad_variance_period | Epoch interval between two gradient variance measurements | [0] | [] | [0] | [0] |
| stgs_grad_variance_samples | Number of STGS gradient samples to use when estimating variance (>=2 enables metric) | [0] | [] | [0] | [0] |
| stgs_hard | Whether to use hard ST-GS | [false] | [] | [false] | [false] |
| teacher_forcing | Use teacher forcing for faster training (single forward pass instead of autoregressive). Changes loss semantics - predicts next token given correct prefix rather than own generations. | [] | [] | [false] | [true] |
| temperature | Temperature for Gumbel-Softmax | [1] | [0.05] | [1] | [100] |