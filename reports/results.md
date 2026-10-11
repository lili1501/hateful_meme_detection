| Run | n | AUROC (95% CI) | Acc @0.5 | Acc @val-tuned thr |
|---|---|---|---|---|
| zero_shot | 398 | 0.5961 (0.5446-0.6511) | 0.5402 | 0.5578 (thr 0.743) |
| lora | 398 | 0.7806 (0.7303-0.8236) | 0.7060 | 0.6633 (thr 0.622) |

AUROC lora - zero_shot: +0.1845 (paired bootstrap 95% CI +0.1303 to +0.2338)
