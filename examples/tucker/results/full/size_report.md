# Model-size study (Tucker-compressed history vs exact checkpointing)

| run | cells (M, with CPML) | nt | default history (GiB) | GPU peak alloc / reserved (GiB) | host peak (GiB) | time per gradient (s) | e_g eps | e_g sigma | max frame error |
|---|---|---|---|---|---|---|---|---|---|
| Z1_checkpoint_opt | 2.4 | 600 | 16 | 2.41 / 2.81 | 0.6 | 1.0 | 2.26e-07 | 3.22e-07 | - |
| Z1_tucker_r16 | 2.4 | 600 | 16 | 0.44 / 0.55 | 0.7 | 2.4 | 8.65e-04 | 4.29e-04 | 0.0152 |
| Z1_tucker_r24 | 2.4 | 600 | 16 | 0.54 / 0.64 | 0.7 | 2.9 | 1.67e-04 | 1.03e-04 | 3.01e-03 |
| Z1_tucker_r32 | 2.4 | 600 | 16 | 0.71 / 0.82 | 0.7 | 3.6 | 2.44e-05 | 2.32e-05 | 4.58e-04 |
| Z1.5_checkpoint_opt | 6.6 | 900 | 67 | 7.76 / 9.54 | 0.6 | 4.4 | - | - | - |
| Z1.5_tucker_r16 | 6.6 | 900 | 67 | 1.13 / 1.36 | 0.7 | 7.3 | 7.11e-04 | 5.80e-04 | 0.041 |
| Z1.5_tucker_r24 | 6.6 | 900 | 67 | 1.29 / 1.54 | 0.7 | 8.2 | 1.59e-04 | 2.93e-04 | 0.0159 |
| Z1.5_tucker_r32 | 6.6 | 900 | 67 | 1.54 / 1.74 | 0.7 | 9.6 | 6.08e-05 | 1.28e-04 | 5.16e-03 |
| Z2_checkpoint_opt | 14.3 | 1200 | 192 | 18.56 / 23.06 | 0.6 | 10.6 | - | - | - |
| Z2_tucker_r16 | 14.3 | 1200 | 192 | 2.30 / 2.83 | 0.7 | 18.4 | 4.43e-04 | 7.16e-04 | 0.0725 |
| Z2_tucker_r24 | 14.3 | 1200 | 192 | 2.53 / 3.03 | 0.7 | 18.7 | 1.40e-04 | 4.67e-04 | 0.0368 |
| Z2_tucker_r32 | 14.3 | 1200 | 192 | 2.89 / 3.45 | 0.7 | 20.5 | 6.55e-05 | 2.16e-04 | 0.0176 |
| Z2.5_offload_checkpoint | 26.4 | 1500 | 442 | 15.40 / 22.08 | 25.7 | 36.7 | - | - | - |
| Z2.5_tucker_r16 | 26.4 | 1500 | 442 | 4.10 / 5.09 | 0.8 | 38.3 | 3.47e-04 | 7.29e-04 | 0.101 |
| Z2.5_tucker_r24 | 26.4 | 1500 | 442 | 4.42 / 5.37 | 0.8 | 40.5 | 1.17e-04 | 3.88e-04 | 0.0577 |
| Z2.5_tucker_r32 | 26.4 | 1500 | 442 | 4.88 / 5.81 | 0.8 | 42.6 | 6.05e-05 | 2.12e-04 | 0.0329 |
| Z3_offload_checkpoint | 43.8 | 1800 | 881 | 17.65 / 19.22 | 82.3 | 93.0 | - | - | - |
| Z3_tucker_r16 | 43.8 | 1800 | 881 | 6.66 / 8.26 | 1.2 | 73.7 | 2.40e-04 | 5.66e-04 | 0.118 |
| Z3_tucker_r24 | 43.8 | 1800 | 881 | 7.06 / 8.66 | 1.2 | 78.2 | 1.03e-04 | 3.50e-04 | 0.078 |
| Z3_tucker_r32 | 43.8 | 1800 | 881 | 7.66 / 9.07 | 1.2 | 81.3 | 4.85e-05 | 2.70e-04 | 0.0498 |
| Z3.5_reference | 67.6 | 2100 | 1586 | skipped: single-level checkpoint needs ~130 GB on the GPU; host-offloaded needs ~533 GB host memory with 8-step segments (limit 150) | | | | | |
| Z3.5_tucker_r16 | 67.6 | 2100 | 1586 | 10.13 / 11.77 | 1.6 | 136.6 | - | - | 0.136 |
| Z3.5_tucker_r24 | 67.6 | 2100 | 1586 | 10.63 / 12.32 | 1.6 | 141.7 | - | - | 0.0915 |
| Z3.5_tucker_r32 | 67.6 | 2100 | 1586 | 11.35 / 13.05 | 1.6 | 149.0 | - | - | 0.0635 |
| Z4_reference | 98.7 | 2400 | 2648 | skipped: single-level checkpoint needs ~200 GB on the GPU; host-offloaded needs ~1777 GB host memory with 4-step segments (limit 150) | | | | | |
| Z4_tucker_r16 | 98.7 | 2400 | 2648 | 14.63 / 17.12 | 1.4 | 231.1 | - | - | - |
| Z4_tucker_r24 | 98.7 | 2400 | 2648 | 15.24 / 17.83 | 1.4 | 242.0 | - | - | - |
| Z4_tucker_r32 | 98.7 | 2400 | 2648 | 16.11 / 18.66 | 1.4 | 249.9 | - | - | - |
| Z4.25_reference | 117.5 | 2550 | 3349 | skipped: single-level checkpoint needs ~245 GB on the GPU | | | | | |
| Z4.25_tucker_r16 | 117.5 | 2550 | 3349 | 13.75 / 16.73 | 1.6 | 317.2 | - | - | - |
| Z4.25_tucker_r24 | 117.5 | 2550 | 3349 | 14.40 / 17.43 | 1.6 | 321.3 | - | - | - |
| Z4.25_tucker_r32 | 117.5 | 2550 | 3349 | 15.31 / 18.25 | 1.6 | 335.5 | - | - | - |
| Z4.5_reference | 138.2 | 2700 | 4171 | skipped: single-level checkpoint needs ~295 GB on the GPU | | | | | |
| Z4.5_tucker_r16 | 138.2 | 2700 | 4171 | 16.12 / 20.79 | 1.8 | 398.2 | - | - | - |
| Z4.5_tucker_r24 | 138.2 | 2700 | 4171 | 16.82 / 21.57 | 1.8 | 412.5 | - | - | - |
| Z4.5_tucker_r32 | 138.2 | 2700 | 4171 | 17.80 / 22.78 | 1.8 | 416.6 | - | - | - |
| Z5_reference | 187.0 | 3000 | 6271 | skipped: single-level checkpoint needs ~420 GB on the GPU | | | | | |
| Z5_tucker_r16 | 187.0 | 3000 | 6271 | failed: torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.20 GiB (GPU 0; 23.65 GiB total  | | | | | |
| Z5_tucker_r24 | 187.0 | 3000 | 6271 | failed: torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.20 GiB (GPU 0; 23.65 GiB total  | | | | | |
| Z5_tucker_r32 | 187.0 | 3000 | 6271 | failed: torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.20 GiB (GPU 0; 23.65 GiB total  | | | | | |

## Dense FWI iterations with the compressed history

| run | cells (M, with CPML) | nt | GPU peak alloc / reserved (GiB) | host peak (GiB) | s per iteration | loss | eps RMSE |
|---|---|---|---|---|---|---|---|
| F3.5_tucker_r16 | 67.6 | 2100 | 10.23 / 10.71 | 1.6 | 151 | 0.0895 -> 0.0752 -> 0.0628 | 0.276 -> 0.276 |
| F4.5_tucker_r16 | 138.2 | 2700 | 20.68 / 23.10 | 1.8 | 398 | 0.0878 -> 0.0738 -> 0.0616 | 0.227 -> 0.227 |
