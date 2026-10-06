# DeepGPR 像素风宣传片 · 源文件

成片：`DeepGPR_pixel_promo_zh.mp4`（中文字幕）/ `DeepGPR_pixel_promo_en.mp4`（英文字幕），1920×1080，30 fps，50 s。
画面在 384×216 的索引色画布上逐帧绘制（固定 26 色调色板），再整数放大 5 倍；配乐为 NumPy 合成的原创 8-bit 音乐。

## 素材来源

所有"数据画面"都由 DeepGPR 自带的 CPU 原生内核（`src/DeepGPR/lib/deepgpr_cpu.so`，`forward` / `backward`）实际计算：

| 镜头 | 计算 | 脚本入口 |
| --- | --- | --- |
| 地下波场（3 炮） | 384×162 网格，dx = 1 cm，600 MHz Ricker，Ez-TM，CPML | `python sims.py scene` |
| B-scan 雷达剖面 | 同一模型 2 cm 网格，95 道共偏移距 | `python sims.py bscan` |
| 梯度 / 双参数 FWI | 128×72 网格，dx = 5 cm，200 MHz，10 炮 × 128 道；流程照 `examples/2.2DFWI.ipynb`（顶部激发、底部接收，Adam，L1 + TV，先 εr 后 εr+σ） | `python sims.py fwi 1 40 110 10` |
| 3D 立方体 | 72³ 网格，dx = 2 cm，400 MHz，Ex 偶极子，mode 3 | `python sims.py cube` |

制作环境装不了 PyTorch，所以 `dgpr_np.py` 把 `DeepGPR.compute` 的 Python 外层（CPML 延拓与系数、状态分配、参数编组）
按原样移植成 NumPy，再用 ctypes 直接调用原生内核；求解器与离散伴随本身未改动。移植后做过核对：伴随梯度与中心差分一致
（3 个测试点相对误差 < 1%），分段推进与一次推进逐位相同。

## 重新渲染

```bash
export OMP_NUM_THREADS=8          # 任意
python sims.py scene && python sims.py bscan && python sims.py cube && python sims.py fwi 1 40 110 10
python audio.py                   # -> music.wav
python render.py video zh out.mp4 # 或 en
ffmpeg -i out.mp4 -i music.wav -c:v copy -c:a aac -b:a 192k -shortest final.mp4
```

依赖：numpy、scipy、pillow、ffmpeg、GNU Unifont（`/usr/share/fonts/opentype/unifont/unifont.otf`，路径在 `gfx.py`）。
`dgpr_np.py` 默认在 `../../src/DeepGPR` 找内核，也可用环境变量 `DEEPGPR_SRC` 指定。
文案集中在 `render.py` 顶部的 `TXT` 字典；分镜时间轴在文件末尾的 `SHOTS`。
