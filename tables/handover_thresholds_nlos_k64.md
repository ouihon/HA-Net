# 3GPP Handover Threshold Comparison in NLOS ($K=64$)

## Threshold = -135 dBm

| Metric          | GT (Unconstrained) | GT (3GPP HO Rule) | Baseline | Ours   | Transformer | xLSTM  |
|-----------------|-------------------:|------------------:|---------:|-------:|------------:|-------:|
| Avg Rate (Mbps) | 458.77             | 285.74            | 148.57   | 270.09 | 229.18      | 229.24 |
| RMSE            | 0.00               | 285.70            | 417.78   | 302.88 | 344.46      | 344.40 |
| HO Total        | 0                  | 23                | 2130     | 24     | 33          | 33     |

## Threshold = -130 dBm

| Metric          | GT (Unconstrained) | GT (3GPP HO Rule) | Baseline | Ours   | Transformer | xLSTM  |
|-----------------|-------------------:|------------------:|---------:|-------:|------------:|-------:|
| Avg Rate (Mbps) | 458.77             | 328.32            | 155.91   | 292.10 | 284.40      | 278.18 |
| RMSE            | 0.00               | 244.85            | 412.45   | 281.42 | 299.28      | 305.04 |
| HO Total        | 0                  | 61                | 5007     | 32     | 125         | 125    |

## Threshold = -125 dBm

| Metric          | GT (Unconstrained) | GT (3GPP HO Rule) | Baseline | Ours   | Transformer | xLSTM  |
|-----------------|-------------------:|------------------:|---------:|-------:|------------:|-------:|
| Avg Rate (Mbps) | 458.77             | 364.69            | 149.21   | 317.71 | 316.56      | 313.57 |
| RMSE            | 0.00               | 209.31            | 417.25   | 248.75 | 268.20      | 270.43 |
| HO Total        | 0                  | 216               | 9242     | 94     | 458         | 461    |
