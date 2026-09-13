# Paper 1 Matched Evaluation Matrix

Accuracy is final-answer correctness after the declared route. A partial cell is
progress only and is not a publishable comparison. `-` means not run or not copied back.

| Dataset or intervention | Direct | ASL base zero-shot (no LoRA) | ASL LoRA GSM-U2000 | ASL LoRA GSM->C* | ASL LoRA target transfer | ASL LoRA target fresh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K official | 61.6% (154/250; N=250) | 0.0% (0/250; N=250) | 44.0% (110/250; N=250) | 32.4% (81/250; N=250) | - | - |
| GSM8K scale x1 | 65.5% (36/55; N=55) | - | 49.1% (27/55; N=55) | - | - | - |
| GSM8K scale x100 | 56.4% (31/55; N=55) | - | 45.5% (25/55; N=55) | - | - | - |
| GSM8K scale x1000 | 43.6% (24/55; N=55) | - | 45.5% (25/55; N=55) | - | - | - |
| GSM8K scale x10000 | 36.4% (20/55; N=55) | - | 43.6% (24/55; N=55) | - | - | - |
| GSM8K scale x1000000 | 30.9% (17/55; N=55) | - | 40.0% (22/55; N=55) | - | - | - |
| GSM8K + operator O1 | 51.0% (51/100; N=100) | 0.0% (0/100; N=100) | - | - | - | - |
| GSM8K + operator O3 | - | - | - | - | - | - |
| GSM8K + operator O5 | - | - | - | - | - | - |
| GSM8K + operator O6 | - | - | - | - | - | - |
| Synthetic depth C1 [diagnostic only] | 100.0% (100/100; N=100) | - | 100.0% (3/3 partial; N=100) | - | - | - |
| Synthetic depth C2 [diagnostic only] | 93.0% (93/100; N=100) | - | 100.0% (100/100; N=100) | 100.0% (100/100; N=100) | - | - |
| Synthetic depth C3 [diagnostic only] | 92.0% (92/100; N=100) | - | - | 100.0% (100/100; N=100) | - | - |
| Synthetic depth C4 [diagnostic only] | 64.0% (64/100; N=100) | - | - | 100.0% (100/100; N=100) | - | - |
| GSM8K jitter +/-30%, x1, O0 | 54.0% (54/100; N=100) | - | 45.0% (45/100; N=100) | - | - | - |
| GSM8K jitter +/-30%, x1000, O0 | 27.5% (11/40 partial; N=100) | - | 42.0% (42/100; N=100) | - | - | - |
| GSM8K jitter +/-30%, x1000, O5 | 11.0% (11/100; N=100) | - | 32.0% (32/100; N=100) | - | - | - |
| SVAMP | - | 0.0% (0/250; N=250) | 65.2% (163/250; N=250) | 54.0% (135/250; N=250) | - | - |
| MAWPS | - | 0.0% (0/250; N=250) | 80.8% (202/250; N=250) | - | - | - |
| ASDiv | 68.0% (170/250; N=250) | 0.0% (0/250; N=250) | 42.8% (107/250; N=250) | - | - | - |
| GSM-Plus | - | 0.0% (0/250; N=250) | 35.6% (89/250; N=250) | - | - | - |

## Notes

- **Synthetic depth C1:** Surface-held-out only; every test graph schema occurs in training.
- **Synthetic depth C2:** Surface-held-out only; every test graph schema occurs in training.
- **Synthetic depth C3:** Surface-held-out only; every test graph schema occurs in training.
- **Synthetic depth C4:** Surface-held-out only; every test graph schema occurs in training.

## Coverage Audit

| Dataset or intervention | Eval SHA-256 | Complete cells | Partial cells |
| --- | --- | ---: | ---: |
| GSM8K official | `3314a09bcd409d66f5bcd3dfe536cf0dc2d7e6ac8f106f876129d2144a6dd0e9` | 4 | 0 |
| GSM8K scale x1 | `0b3fc17242b1581db25fd517f51e2e64a20b3abe7ea5b3e8ed36339b91c6060d` | 2 | 0 |
| GSM8K scale x100 | `4d5809d4ebb97cc8aa2a93a9db17c58ac32719cf3d0670b1d668d78b59d3ff59` | 2 | 0 |
| GSM8K scale x1000 | `96dae9671b4c10d9ea2c3b42c3278928763341a9b697cc16c7b16c0380217b7c` | 2 | 0 |
| GSM8K scale x10000 | `9edcf3953fb96c651ac944bb4e8d15d60f05123cb86c2cfa92c55b4250155d61` | 2 | 0 |
| GSM8K scale x1000000 | `49cd9aaf823a6f1c1446ff16674117143871d1fe43a17addfffc691f0d4b5001` | 2 | 0 |
| GSM8K + operator O1 | `2c252fedd30b01b800a2e2a37a9a4c871972f44a0d439cadf056624549b3d6bf` | 2 | 0 |
| GSM8K + operator O3 | `a5d4bbae658cd7be808e9cdd9dc9324ead67adf9e054c6637c0cca9d26e463de` | 0 | 0 |
| GSM8K + operator O5 | `195d1ff0b4b28735b2bada2c5ac689baf9eaeec850116223fa085a93c27844f7` | 0 | 0 |
| GSM8K + operator O6 | `498e2ece0a242c0a89c186f191dcbe22aa1d14a0ed3642b2b56e536e2c1c4a83` | 0 | 0 |
| Synthetic depth C1 | `111948098fadb576709b7fb3b9673837d3bbba13957a502d494a5b6c507593ef` | 1 | 1 |
| Synthetic depth C2 | `dce27f31ff83f6e904030101c739823d613f0c2a674d7a51488062572d7bce97` | 3 | 0 |
| Synthetic depth C3 | `0462cad01c18a52465a1707a1b7ca08cf431d4895f88e4608f9e0142e3e52c31` | 2 | 0 |
| Synthetic depth C4 | `64d308e936334a59f243fde13508f65e3adfa137176199e276a47ad66a3464ce` | 2 | 0 |
| GSM8K jitter +/-30%, x1, O0 | `4e3fc5b39487c5dae27757456f83047fd412b14aa1120396591640b1e4b659f2` | 2 | 0 |
| GSM8K jitter +/-30%, x1000, O0 | `f65c1acc5ec7716dbd25c0c5febd33116ad34ea4d951c3b7d946195201b2f7f3` | 1 | 1 |
| GSM8K jitter +/-30%, x1000, O5 | `df3920bfed0f7c3b5aa8c76412239e82bff124ce350066680244c220615c2658` | 2 | 0 |
| SVAMP | `4dab3b3ed28412d4f34ddb836089c5214c2d54a35d924149c17c203daffa44b0` | 3 | 0 |
| MAWPS | `79426260e89a21214ab2c714c1d44aacc0fc0755d0600b339b9783bc9690e973` | 2 | 0 |
| ASDiv | `74a0397971c2359b2626d54b1b0dd5b79ef81e42403c5755fccf58e92e0ab737` | 3 | 0 |
| GSM-Plus | `d6a380071c0577f4f50e714ae3a2dab92d5af974e4c2624adde32148a6ddafd9` | 2 | 0 |
