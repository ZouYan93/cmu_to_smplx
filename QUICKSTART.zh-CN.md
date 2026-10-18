# CMU 转 SMPL-X

运行环境：Python 3.10+、NumPy、SciPy。

```bash
python -m pip install .
cmu-to-smplx captures/02_01.bvh --asf captures/02.asf --model models/SMPLX_NEUTRAL.npz --fps 120 -o output/02_01.npz
```

输入由用户持有：完整 31 关节 cgspeed BVH、原始校姿帧、同组官方 ASF，
以及获得使用授权的 `SMPLX_NEUTRAL.npz`。模型版本哈希见英文 README。

`--fps` 填写原始动作帧率。也可用 `--timing-manifest` 提供哈希绑定的计时清单。
BVH 文件头的 30 Hz 字段仅作为来源信息记录。

输出为 `.npz` 和同名 `.json`。动作帧逐帧保留，移除一行校姿数据；
目标体形为 neutral、16 维零值。包含肘腕语义映射，手指与面部未估计。
平滑、动作拟合、修复和重采样均未执行。

包内仅含代码与离线学习的 22 个旋转矩阵。ASF、人体模型与采集文件由用户另行持有。
