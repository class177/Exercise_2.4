(a) Use QuaDRiGa to generate real MIMO channels for one specific configuration.
# Exercise 2.4(a) – QuaDRiGa MIMO 通道資料集產生

本專案示範如何使用 **QuaDRiGa** 產生一組寫實的 **2×4 MIMO** 無線通道資料集，場景採用 3GPP TR 38.901 中的 **UMi NLOS**（都市微小區、非視線）設定。產生的資料將作為後續練習（例如 Channel GAN）之訓練用資料。

---

## 📁 專案內容概要

本練習的目標是透過 QuaDRiGa 產生一個具有下列特性的 MIMO 通道資料集：

- 場景：`3GPP_38.901_UMi_NLOS`
- 頻率：3.5 GHz
- 發射天線數（BS）：4
- 接收天線數（UE）：2
- UE 速度：3 km/h
- 通道快照數：20,000
- BS 位置：`[0; 0; 25]`（m）
- UE 初始位置：`[100; 0; 1.5]`（m）
- UE 軌跡：直線軌跡（linear track）

輸出檔案為 **`mimo_channel_dataset.mat`**，內含：

- `h_coeff`：原始多徑 MIMO 通道張量
- `h_mimo`：對多徑加總後之平坦衰落 MIMO 通道

---

## 🧩 系統設定與參數

| 參數項目              | 值/說明                    |
|-----------------------|---------------------------|
| 場景 Scenario         | 3GPP_38.901_UMi_NLOS      |
| 載波頻率              | 3.5 GHz                   |
| Tx 天線數（BS）       | 4                         |
| Rx 天線數（UE）       | 2                         |
| UE 速度               | 3 km/h                    |
| 通道快照數 Nsnap      | 20,000                    |
| BS 位置               | [0; 0; 25] m              |
| UE 初始位置           | [100; 0; 1.5] m           |
| UE 軌跡               | 線性軌跡 linear track     |

通道張量尺寸：

- 原始多徑通道 `h_coeff`：
  - 尺寸：`[2, 4, 58, 20000]`
  - 意義：
    - `Nr = 2`：接收天線數
    - `Nt = 4`：發射天線數
    - `Npath = 58`：多徑路徑數
    - `Nsnap = 20000`：快照數

- 平坦衰落通道 `h_mimo`：
  - 尺寸：`[2, 4, 20000]`
  - 代表每個 snapshot 對應一個 2×4 的等效 MIMO 通道矩陣

---

## ⚙️ 執行流程概述

完整實作在 `Exercise_2_4a.m` 中，主要步驟如下：

1. **加入 QuaDRiGa 路徑**
   - 將 `QuaDRiGa-main` 及其子資料夾加入 MATLAB path。
   - 確認以下類別可以成功呼叫：
     - `qd_layout`
     - `qd_simulation_parameters`
     - `qd_arrayant`

2. **建立模擬參數**
   - 建立 `qd_simulation_parameters` 物件。
   - 設定中心頻率為 3.5 GHz。

3. **建立 Layout 與天線陣列**
   - 使用 `qd_layout` 建立場景佈局。
   - 設定 BS 為 4 根 Tx 天線、UE 為 2 根 Rx 天線。
   - 設定 BS/UE 的空間位置與高度。

4. **設定 UE 軌跡與移動**
   - 為 UE 指定起點位置與直線移動路徑。
   - UE 速度設定為 3 km/h，讓 QuaDRiGa 沿途生成通道。

5. **選擇場景**
   - 指定通道場景為 `3GPP_38.901_UMi_NLOS`。

6. **產生通道係數**
   - 依序呼叫：
     - `init_builder`
     - `gen_parameters`
     - `get_channels`
   - 得到原始多徑通道張量 `h_coeff`（`2 x 4 x 58 x 20000`）。

7. **轉換為平坦衰落通道**
   - 在多徑維度（第三維，大小 58）上做加總：
     - `h_mimo = sum(h_coeff, dim = 3)`
   - 得到平坦衰落通道 `h_mimo`（`2 x 4 x 20000`）。

8. **儲存結果**
   - 將 `h_coeff` 與 `h_mimo` 一併存為：
     - `mimo_channel_dataset.mat`

---

## 📦 輸出檔案說明

### `mimo_channel_dataset.mat`



(b) Use the GAN-based framework to model the MIMO channel you generated in 2.4(a).
training by colab:https://github.com/class177/Exercise_2.4/blob/main/HW2_4.ipynb

result image out:(b)result image(ChannelGAN_MIMO) folder

console out:
Loaded h_dataset shape = (2, 4, 20000)
data shape        = (10000, 4)
conditioning shape = (10000, 24)
Iter:    100 | D_loss:   1.7816 | G_loss:  -0.9656
Iter:    200 | D_loss:   1.2791 | G_loss:  -0.3240
Iter:    300 | D_loss:  -0.5191 | G_loss:   0.7665
Iter:    400 | D_loss:   1.2809 | G_loss:  -0.2930
Iter:    500 | D_loss:   0.0227 | G_loss:   0.5791
Iter:    600 | D_loss:  -0.1493 | G_loss:   0.2786
Iter:    700 | D_loss:  -0.1214 | G_loss:   0.3006
Iter:    800 | D_loss:  -0.0760 | G_loss:   0.2091
Iter:    900 | D_loss:  -0.0247 | G_loss:   0.1679
Iter:   1000 | D_loss:   0.0247 | G_loss:   0.1244
Model saved to: ./Models/ChannelGAN_MIMO_step_1000.ckpt
Iter:   1100 | D_loss:   0.0308 | G_loss:   0.2234
Iter:   1200 | D_loss:   0.0401 | G_loss:  -0.2089
Iter:   1300 | D_loss:   0.0273 | G_loss:  -0.4249
Iter:   1400 | D_loss:   0.0405 | G_loss:  -0.5876
Iter:   1500 | D_loss:   0.0247 | G_loss:  -0.4846
Iter:   1600 | D_loss:   0.0664 | G_loss:  -0.4558
Iter:   1700 | D_loss:   0.0317 | G_loss:  -0.6353
Iter:   1800 | D_loss:   0.0355 | G_loss:  -0.6995
Iter:   1900 | D_loss:   0.0364 | G_loss:  -0.6310
Iter:   2000 | D_loss:   0.0261 | G_loss:  -0.6890
Model saved to: ./Models/ChannelGAN_MIMO_step_2000.ckpt
Iter:   2100 | D_loss:   0.0459 | G_loss:  -0.5405
Iter:   2200 | D_loss:   0.0356 | G_loss:  -0.3620
Iter:   2300 | D_loss:   0.0390 | G_loss:  -0.3980
Iter:   2400 | D_loss:   0.0451 | G_loss:  -0.2548
Iter:   2500 | D_loss:   0.0327 | G_loss:  -0.2829
Iter:   2600 | D_loss:   0.0459 | G_loss:  -0.2378
Iter:   2700 | D_loss:   0.0364 | G_loss:  -0.3374
Iter:   2800 | D_loss:   0.0440 | G_loss:  -0.1684
Iter:   2900 | D_loss:   0.0326 | G_loss:  -0.1191
Iter:   3000 | D_loss:   0.0531 | G_loss:  -0.0601
Model saved to: ./Models/ChannelGAN_MIMO_step_3000.ckpt
Iter:   3100 | D_loss:   0.0458 | G_loss:   0.0370
Iter:   3200 | D_loss:   0.0468 | G_loss:  -0.0642
Iter:   3300 | D_loss:   0.0244 | G_loss:  -0.2826
Iter:   3400 | D_loss:   0.0142 | G_loss:  -0.2067
Iter:   3500 | D_loss:   0.0173 | G_loss:  -0.2773
Iter:   3600 | D_loss:   0.0250 | G_loss:  -0.1054
Iter:   3700 | D_loss:   0.0656 | G_loss:   0.0417
Iter:   3800 | D_loss:   0.0166 | G_loss:  -0.2296
Iter:   3900 | D_loss:   0.0215 | G_loss:  -0.0052
Iter:   4000 | D_loss:   0.0431 | G_loss:  -0.1284
Model saved to: ./Models/ChannelGAN_MIMO_step_4000.ckpt
Iter:   4100 | D_loss:   0.0214 | G_loss:   0.3647
Iter:   4200 | D_loss:   0.0278 | G_loss:   0.5917
Iter:   4300 | D_loss:   0.0285 | G_loss:   0.2825
Iter:   4400 | D_loss:   0.0481 | G_loss:   0.1369
Iter:   4500 | D_loss:   0.0397 | G_loss:   0.0182
Iter:   4600 | D_loss:   0.0523 | G_loss:  -0.1024
Iter:   4700 | D_loss:   0.0486 | G_loss:  -0.0937
Iter:   4800 | D_loss:   0.0349 | G_loss:  -0.1540
Iter:   4900 | D_loss:   0.0456 | G_loss:  -0.0242
Iter:   5000 | D_loss:   0.0189 | G_loss:   0.3342
Model saved to: ./Models/ChannelGAN_MIMO_step_5000.ckpt
Iter:   5100 | D_loss:   0.0011 | G_loss:   0.0528
Iter:   5200 | D_loss:   0.0477 | G_loss:   0.0190
Iter:   5300 | D_loss:   0.0194 | G_loss:   0.2410
Iter:   5400 | D_loss:   0.0363 | G_loss:  -0.2699
Iter:   5500 | D_loss:   0.0512 | G_loss:  -0.3813
Iter:   5600 | D_loss:   0.0476 | G_loss:  -0.3042
Iter:   5700 | D_loss:   0.0296 | G_loss:  -0.3352
Iter:   5800 | D_loss:   0.0331 | G_loss:  -0.3226
Iter:   5900 | D_loss:   0.0109 | G_loss:  -0.0712
Iter:   6000 | D_loss:   0.0429 | G_loss:  -0.0503
Model saved to: ./Models/ChannelGAN_MIMO_step_6000.ckpt
Iter:   6100 | D_loss:   0.0113 | G_loss:   0.0119
Iter:   6200 | D_loss:   0.0327 | G_loss:  -0.1103
Iter:   6300 | D_loss:   0.0193 | G_loss:  -0.1777
Iter:   6400 | D_loss:   0.0139 | G_loss:  -0.4839
Iter:   6500 | D_loss:   0.0239 | G_loss:   0.3622
Iter:   6600 | D_loss:   0.0202 | G_loss:  -0.2398
Iter:   6700 | D_loss:   0.0458 | G_loss:  -0.4861
Iter:   6800 | D_loss:   0.0158 | G_loss:  -0.7977
Iter:   6900 | D_loss:   0.0185 | G_loss:  -0.7885
Iter:   7000 | D_loss:   0.0375 | G_loss:  -0.6831
Model saved to: ./Models/ChannelGAN_MIMO_step_7000.ckpt
Iter:   7100 | D_loss:   0.0279 | G_loss:  -1.3474
Iter:   7200 | D_loss:   0.0856 | G_loss:  -1.4322
Iter:   7300 | D_loss:   0.0638 | G_loss:  -1.1416
Iter:   7400 | D_loss:   0.0521 | G_loss:  -1.1447
Iter:   7500 | D_loss:   0.0758 | G_loss:  -1.0456
Iter:   7600 | D_loss:   0.0249 | G_loss:  -0.7039
Iter:   7700 | D_loss:   0.0154 | G_loss:  -0.4246
Iter:   7800 | D_loss:   0.0505 | G_loss:  -1.2051
Iter:   7900 | D_loss:   0.0118 | G_loss:  -0.9629
Iter:   8000 | D_loss:   0.0393 | G_loss:  -1.2253
Model saved to: ./Models/ChannelGAN_MIMO_step_8000.ckpt
Iter:   8100 | D_loss:   0.0227 | G_loss:  -0.9989
Iter:   8200 | D_loss:   0.0364 | G_loss:  -1.4542
Iter:   8300 | D_loss:   0.0538 | G_loss:  -1.7361
Iter:   8400 | D_loss:   0.0049 | G_loss:  -0.8254
Iter:   8500 | D_loss:   0.0148 | G_loss:  -1.0897
Iter:   8600 | D_loss:   0.0223 | G_loss:  -0.9216
Iter:   8700 | D_loss:   0.0304 | G_loss:  -0.8574
Iter:   8800 | D_loss:   0.0183 | G_loss:  -0.8739
Iter:   8900 | D_loss:   0.0247 | G_loss:  -1.0559
Iter:   9000 | D_loss:   0.0512 | G_loss:  -0.6129
Model saved to: ./Models/ChannelGAN_MIMO_step_9000.ckpt
Iter:   9100 | D_loss:   0.0626 | G_loss:  -1.2564
Iter:   9200 | D_loss:   0.0407 | G_loss:  -1.2348
Iter:   9300 | D_loss:  -0.0362 | G_loss:  -1.0050
Iter:   9400 | D_loss:   0.0284 | G_loss:  -1.6845
Iter:   9500 | D_loss:   0.0809 | G_loss:  -2.0131
Iter:   9600 | D_loss:   0.0066 | G_loss:  -1.1035
Iter:   9700 | D_loss:   0.0181 | G_loss:  -0.4184
Iter:   9800 | D_loss:   0.0124 | G_loss:  -1.2307
Iter:   9900 | D_loss:   0.0595 | G_loss:  -1.2082
Iter:  10000 | D_loss:   0.0371 | G_loss:  -0.8976
Model saved to: ./Models/ChannelGAN_MIMO_step_10000.ckpt
Final model saved to: ./Models/ChannelGAN_MIMO_final.ckpt
