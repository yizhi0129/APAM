# caractéristique des GPUs disponibles

`nvidia-smi` donne des informations des GPUs, par exemple :
```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 3080    Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   55C    P2    80W / 320W |   500MiB / 10000MiB  |     25%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      4567      C   ./my_program                   450MiB    |
|    0   N/A  N/A      7891      G   /usr/bin/X                        50MiB   |
+-----------------------------------------------------------------------------+
```

