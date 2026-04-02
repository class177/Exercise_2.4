# -*- coding: utf-8 -*-
"""
Exercise 2.4(b): MIMO Channel GAN Implementation

此程式使用條件式 WGAN-GP 來學習在 Exercise 2.4(a) 所產生的 MIMO 通道模型。
通道資料來自 QuaDRiGa 產生之 h_mimo (2 x 4 x Nsnap)。

主要流程：
    1. 從 h_mimo 中隨機取一個通道快照 H
    2. 隨機產生一個 16-QAM 發射向量 x
    3. 模擬 y = H x + n
    4. 將 y 的實部與虛部當作「真實樣本」(real sample)
    5. 將 x 與 H 的實部/虛部拼接成「條件向量」(conditioning vector)
    6. 使用條件式 WGAN-GP 訓練，使 G 在給定 (x, H) 的條件下生成逼真的 y
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib

# 不開圖形視窗，將圖直接存檔
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 使用 TF1 風格 API
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# 綁定 GPU 設定（在本機多 GPU 環境有用；在 Colab 可以註解掉這兩行）
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# 若想讓系統自選 GPU 或使用 CPU，可改成註解掉下一行
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 固定亂數種子以便重現結果
tf.set_random_seed(100)
np.random.seed(100)

# 檔案與訓練設定
MAT_FILE_PATH   = "mimo_channel_dataset.mat"  # 由 Exercise 2.4(a) 生成
MODEL_NAME      = "ChannelGAN_MIMO"
SAVE_FIG_PATH   = MODEL_NAME + "_images"      # 圖片輸出目錄
SAVE_MODEL_PATH = "./Models"                  # 模型 checkpoint 目錄

DATA_SIZE   = 10000    # 欲從通道資料中產生多少 (x, H, y) 樣本
BATCH_SIZE  = 512
Z_DIM       = 16       # Generator 噪音維度
TRAIN_ITERS = 10000    # 總訓練 iteration 次數
D_STEPS     = 10       # 每次更新 G 前，先更新 D 的次數
PLOT_EVERY  = 1000     # 每多少 iteration 存模型並畫圖
NOISE_VAR   = 0.01     # AWGN 雜訊變異數 (每實/虛部)

# MIMO 天線維度（需與 h_mimo 一致）
NUM_RX = 2  # 接收天線數 Nr
NUM_TX = 4  # 發射天線數 Nt

# 網路輸入/輸出維度
# y 的維度：每個 Rx 有實部+虛部 → 2 * Nr
OUTPUT_DIM    = 2 * NUM_RX

# Condition 向量 = Re/Im(x) + Re/Im(H_flat)
# x : Nt 個複數 → 2 * Nt
# H : Nr x Nt 個複數 → 2 * Nr * Nt
CONDITION_DIM = 2 * NUM_TX + 2 * NUM_RX * NUM_TX

# 16-QAM 星座點 (平面 4x4 正方形 QAM)
MEAN_SET_QAM = np.asarray([
    -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
    -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
     1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,
     3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j
], dtype=np.complex64)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def sample_Z(sample_size):
    """
    產生 Generator 的輸入噪音 z
    從標準常態分配 N(0,1) 取樣。

    Args:
        sample_size: tuple, e.g. (batch_size, Z_DIM)

    Returns:
        z: np.ndarray, dtype float32
    """
    return np.random.normal(size=sample_size).astype(np.float32)


def xavier_init(size):
    """
    Xavier 權重初始化。

    Args:
        size: list or tuple, [fan_in, fan_out]

    Returns:
        TensorFlow 隨機張量
    """
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def ensure_dir(path):
    """
    若資料夾 path 不存在，則建立。
    """
    if not os.path.exists(path):
        os.makedirs(path)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def load_mimo_dataset(mat_file_path):
    """
    從 .mat 檔載入 MIMO 通道 h_mimo。

    Expected:
        h_mimo: complex ndarray, shape = [Nr, Nt, Nsnap]

    Returns:
        h_dataset: ndarray, shape [Nr, Nt, Nsnap]
    """
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(
            "找不到檔案 '{}'. 請先執行 Exercise 2.4(a) 產生 "
            "'mimo_channel_dataset.mat'。".format(mat_file_path)
        )

    mat_data = sio.loadmat(mat_file_path)

    if "h_mimo" not in mat_data:
        raise KeyError(
            "檔案 '{}' 中沒有變數 'h_mimo'. "
            "請檢查在 MATLAB 中 save() 的變數名稱。".format(mat_file_path)
        )

    h_dataset = mat_data["h_mimo"]

    if h_dataset.ndim != 3:
        raise ValueError(
            "預期 h_mimo 維度為 3 維 [Nr, Nt, Nsnap]，實際為 {}.".format(h_dataset.shape)
        )

    return h_dataset


def generate_real_samples_with_labels_Rayleigh(
    h_dataset, number=100, noise_var=0.01
):
    """
    使用 QuaDRiGa 的 h_mimo 來產生 GAN 的「真實樣本」(real sample) 及「條件向量」(conditioning)。

    每個樣本的產生流程：
        1) 從 h_dataset 隨機選取一個 snapshot 的 H (Nr x Nt)
        2) 每個 Tx 天線選一個 16-QAM symbol，組成發射向量 x (Nt x 1)
        3) 模擬接收信號 y = H x + n，其中 n 為複數 AWGN
        4) 將 y 的實部、虛部拼成 real sample，大小 = 2 * Nr
        5) 將 x、H 的實部與虛部展平成 condition 向量，大小 = 2*Nt + 2*Nr*Nt

    Args:
        h_dataset: ndarray, shape [Nr, Nt, Nsnap]
        number: 欲產生樣本數量
        noise_var: 複數 AWGN 每實/虛部變異數

    Returns:
        received_data: shape [number, 2*Nr]
        conditioning:  shape [number, 2*Nt + 2*Nr*Nt]
    """
    Nr, Nt, Nsnap = h_dataset.shape

    received_data_list = []
    conditioning_list = []

    for _ in range(number):
        # 1) 從 Nsnap 中隨機選一個通道 snapshot
        idx = np.random.choice(Nsnap)
        H = h_dataset[:, :, idx]  # shape: [Nr, Nt], complex

        # 2) 產生發射向量 x：每根 Tx 天線一個 QAM symbol
        symbol_idx = np.random.choice(len(MEAN_SET_QAM), Nt)
        x = MEAN_SET_QAM[symbol_idx].reshape(Nt, 1)  # shape: [Nt, 1]

        # 3) 模擬接收信號 y = H x + n
        noise = np.sqrt(noise_var / 2.0) * (
            np.random.randn(Nr, 1) + 1j * np.random.randn(Nr, 1)
        )
        y = H @ x + noise  # shape: [Nr, 1]

        # 4) real sample: y 的實部與虛部疊起來
        #    [Re(y1), Re(y2), ..., Im(y1), Im(y2), ...]
        y_vec = np.hstack([
            np.real(y).flatten(),
            np.imag(y).flatten()
        ]).astype(np.float32)

        # 5) condition 向量 = [Re(x), Im(x), Re(H_flat), Im(H_flat)]
        H_flat = H.flatten()
        cond_vec = np.hstack([
            np.real(x).flatten(),
            np.imag(x).flatten(),
            np.real(H_flat).flatten(),
            np.imag(H_flat).flatten()
        ]).astype(np.float32)

        # 為了讓數值較穩定，做一點縮放 (與原始 SISO 版本一致)
        cond_vec = cond_vec / 3.0

        received_data_list.append(y_vec)
        conditioning_list.append(cond_vec)

    # 堆疊成 numpy array
    received_data = np.asarray(received_data_list, dtype=np.float32)
    conditioning  = np.asarray(conditioning_list, dtype=np.float32)

    return received_data, conditioning


# -----------------------------------------------------------------------------
# Model definition: Generator & Discriminator
# -----------------------------------------------------------------------------
def generator_conditional(z, conditioning):
    """
    條件式 Generator：
    輸入：隨機噪音 z 以及條件向量 conditioning
    輸出：生成的 y_hat（與真實 y 向量同維度）

    G 結構：
        [z, cond] -> 全連接 128 -> 128 -> 128 -> 輸出 OUTPUT_DIM
    """
    # 將 z 與 condition 在維度 1 上做拼接: [batch, Z_DIM + CONDITION_DIM]
    z_combine = tf.concat([z, conditioning], axis=1)

    # 三層全連接 + ReLU
    G_h1 = tf.nn.relu(tf.matmul(z_combine, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1,      G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2,      G_W3) + G_b3)

    # 最後一層線性輸出，不加 activation
    G_logit = tf.matmul(G_h3, G_W4) + G_b4
    return G_logit  # shape: [batch, OUTPUT_DIM]


def discriminator_conditional(X, conditioning):
    """
    條件式 Discriminator：
    輸入：樣本 X（真實 y 或生成 y_hat）以及條件向量 conditioning
    輸出：
        D_prob  : sigmoid 後的「為真實樣本的機率」
        D_logit : 未經 sigmoid 的 logits (WGAN 會用)

    D 結構：
        [X, cond] -> 全連接 32 -> 32 -> 32 -> 輸出 1
    """
    # 將輸入樣本與條件向量拼在一起
    x_combine = tf.concat([X, conditioning], axis=1)

    # 原始程式中有除以 4.0，做一點幅度縮放，避免輸入過大
    D_h1 = tf.nn.relu(tf.matmul(x_combine / 4.0, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1,            D_W2) + D_b2)
    D_h3 = tf.nn.relu(tf.matmul(D_h2,            D_W3) + D_b3)
    D_logit = tf.matmul(D_h3, D_W4) + D_b4
    D_prob  = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


# -----------------------------------------------------------------------------
# Build graph (WGAN-GP)
# -----------------------------------------------------------------------------
# Discriminator 參數
D_W1 = tf.Variable(xavier_init([OUTPUT_DIM + CONDITION_DIM, 32]))
D_b1 = tf.Variable(tf.zeros(shape=[32]))
D_W2 = tf.Variable(xavier_init([32, 32]))
D_b2 = tf.Variable(tf.zeros(shape=[32]))
D_W3 = tf.Variable(xavier_init([32, 32]))
D_b3 = tf.Variable(tf.zeros(shape=[32]))
D_W4 = tf.Variable(xavier_init([32, 1]))
D_b4 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3, D_W4, D_b4]

# Generator 參數
G_W1 = tf.Variable(xavier_init([Z_DIM + CONDITION_DIM, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))
G_W2 = tf.Variable(xavier_init([128, 128]))
G_b2 = tf.Variable(tf.zeros(shape=[128]))
G_W3 = tf.Variable(xavier_init([128, 128]))
G_b3 = tf.Variable(tf.zeros(shape=[128]))
G_W4 = tf.Variable(xavier_init([128, OUTPUT_DIM]))
G_b4 = tf.Variable(tf.zeros(shape=[OUTPUT_DIM]))
theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3, G_W4, G_b4]

# Placeholders：輸入真實樣本、噪音 z、條件向量
R_sample  = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
Z         = tf.placeholder(tf.float32, shape=[None, Z_DIM])
Condition = tf.placeholder(tf.float32, shape=[None, CONDITION_DIM])

# 前向傳遞：得到生成樣本與 D 在真實/生成樣本上的輸出
G_sample = generator_conditional(Z, Condition)
D_prob_real, D_logit_real = discriminator_conditional(R_sample, Condition)
D_prob_fake, D_logit_fake = discriminator_conditional(G_sample, Condition)

# --------------------- WGAN-GP 損失函數 ---------------------
# WGAN-value function:
#   D 要最大化：E[D(real)] - E[D(fake)]
#   這裡寫成 minimize，所以我們定義：
#   D_loss = E[D(fake)] - E[D(real)] + λ * GP
D_loss = tf.reduce_mean(D_logit_fake) - tf.reduce_mean(D_logit_real)

# G 要最小化：-E[D(fake)]
G_loss = -tf.reduce_mean(D_logit_fake)

# Gradient penalty 的權重 λ
lambda_gp = 5.0

# WGAN-GP：在 real 與 fake 之間插值，計算 gradient norm
alpha = tf.random_uniform(shape=tf.shape(R_sample), minval=0.0, maxval=1.0)
differences  = G_sample - R_sample
interpolates = R_sample + alpha * differences
_, D_inter   = discriminator_conditional(interpolates, Condition)

# 對 interpolates 計算梯度 ∂D/∂(interpolates)
gradients = tf.gradients(D_inter, [interpolates])[0]

# 每筆樣本的梯度 L2 norm
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)

# Gradient penalty：(‖∇D‖_2 - 1)^2
gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)

# 將 GP 項加到 D_loss
D_loss += lambda_gp * gradient_penalty

# 利用 Adam 做參數更新
D_solver = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.5, beta2=0.9
).minimize(D_loss, var_list=theta_D)

G_solver = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.5, beta2=0.9
).minimize(G_loss, var_list=theta_G)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_real_distribution(data, save_fig_path):
    """
    畫出「真實資料」中第 1 根接收天線 (Rx1) 的散佈圖。
    x 軸：Re(y1)
    y 軸：Im(y1)
    """
    plt.figure(figsize=(5, 5))
    # data 的結構為 [Re(y1), Re(y2), ..., Im(y1), Im(y2), ...]
    plt.plot(data[:1000, 0], data[:1000, NUM_RX], "b.", alpha=0.6)
    plt.xlabel(r"$Re\{y_1\}$")
    plt.ylabel(r"$Im\{y_1\}$")
    plt.title("Real data distribution of Rx antenna 1")
    plt.grid(True, alpha=0.3)
    ensure_dir(save_fig_path)
    plt.savefig(os.path.join(save_fig_path, "real_rx1.png"), bbox_inches="tight")
    plt.close()


def build_plot_conditioning_from_fixed_channel(H, number=20):
    """
    為了畫圖可視化，固定某一個 MIMO 通道 H，
    只改變 Tx1 的 16-QAM symbol (其他 Tx 設為 0)，來看不同輸入 x 下的生成 y 分布。

    Args:
        H: 單一快照的 MIMO 通道矩陣，shape [Nr, Nt]
        number: 每個 QAM symbol 重複生成幾次（搭配不同 z）

    Returns:
        conditioning: ndarray, shape [16*number, CONDITION_DIM]
    """
    H_flat = H.flatten()
    conditioning_list = []

    # 掃過 16QAM 所有 symbol
    for qam_symbol in MEAN_SET_QAM:
        for _ in range(number):
            # 只在第 1 根 Tx 天線上送此 symbol，其餘設為 0
            x = np.zeros((NUM_TX, 1), dtype=np.complex64)
            x[0, 0] = qam_symbol

            cond_vec = np.hstack([
                np.real(x).flatten(),
                np.imag(x).flatten(),
                np.real(H_flat).flatten(),
                np.imag(H_flat).flatten()
            ]).astype(np.float32) / 3.0

            conditioning_list.append(cond_vec)

    return np.asarray(conditioning_list, dtype=np.float32)


def plot_generated_samples(
    sess, G_sample_tensor, Z_tensor, Condition_tensor,
    h_dataset, save_fig_path, step
):
    """
    固定隨機選取的一個 MIMO 通道 H，
    讓 Generator 生成對應之 Rx1 接收信號分布。

    x 軸：Re(y1_hat)
    y 軸：Im(y1_hat)
    """
    # 隨機從 h_dataset 中選一個 snapshot
    idx = np.random.choice(h_dataset.shape[2])
    H = h_dataset[:, :, idx]

    # 生成對應的 conditioning，掃過 QAM symbol
    conditioning = build_plot_conditioning_from_fixed_channel(H, number=20)

    # 產生噪音 z 並丟進 G
    z_input = sample_Z((conditioning.shape[0], Z_DIM))
    samples = sess.run(
        G_sample_tensor,
        feed_dict={Z_tensor: z_input, Condition_tensor: conditioning}
    )

    plt.figure(figsize=(5, 5))
    plt.plot(samples[:, 0], samples[:, NUM_RX], "r.", alpha=0.6)
    plt.xlabel(r"$Re\{\hat{y}_1\}$")
    plt.ylabel(r"$Im\{\hat{y}_1\}$")
    plt.title("Generated samples of Rx antenna 1 at step {}".format(step))
    plt.grid(True, alpha=0.3)
    ensure_dir(save_fig_path)
    plt.savefig(
        os.path.join(save_fig_path, "generated_rx1_step_{:06d}.png".format(step)),
        bbox_inches="tight"
    )
    plt.close()


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def main():
    # 確保圖檔與模型資料夾存在
    ensure_dir(SAVE_FIG_PATH)
    ensure_dir(SAVE_MODEL_PATH)

    # 1. 載入 MIMO 通道資料
    h_dataset = load_mimo_dataset(MAT_FILE_PATH)
    print("Loaded h_dataset shape =", h_dataset.shape)

    # 確認維度與設定一致
    if h_dataset.shape[0] != NUM_RX or h_dataset.shape[1] != NUM_TX:
        raise ValueError(
            "預期 h_dataset 前兩維為 ({}, {}), 實際為 {}.".format(
                NUM_RX, NUM_TX, h_dataset.shape
            )
        )

    # 2. 產生訓練用的 (y, condition) 資料
    data, conditioning_all = generate_real_samples_with_labels_Rayleigh(
        h_dataset, number=DATA_SIZE, noise_var=NOISE_VAR
    )
    print("data shape        =", data.shape)
    print("conditioning shape =", conditioning_all.shape)

    # 畫出真實資料分佈（Rx1）
    plot_real_distribution(data, SAVE_FIG_PATH)

    # 3. 建立 Session 並初始化變數
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # 4. 開始訓練迴圈
    for it in range(TRAIN_ITERS):
        # 用簡單的循環 index 方式取 batch，走完一輪後再重新洗牌
        start_idx = (it * BATCH_SIZE) % DATA_SIZE

        if start_idx + BATCH_SIZE >= len(data):
            # 走完一輪 data，重新打亂
            perm = np.random.permutation(DATA_SIZE)
            data            = data[perm]
            conditioning_all = conditioning_all[perm]
            start_idx = 0

        X_mb   = data[start_idx:start_idx + BATCH_SIZE, :]
        cond_mb = conditioning_all[start_idx:start_idx + BATCH_SIZE, :]

        # (a) 先更新 Discriminator 多次（提升 D 的辨識能力）
        for _ in range(D_STEPS):
            _, D_loss_curr = sess.run(
                [D_solver, D_loss],
                feed_dict={
                    R_sample: X_mb,
                    Z: sample_Z((BATCH_SIZE, Z_DIM)),
                    Condition: cond_mb
                }
            )

        # (b) 接著更新一次 Generator
        _, G_loss_curr = sess.run(
            [G_solver, G_loss],
            feed_dict={
                R_sample: X_mb,
                Z: sample_Z((BATCH_SIZE, Z_DIM)),
                Condition: cond_mb
            }
        )

        # 每 100 iter 印一次 loss
        if (it + 1) % 100 == 0:
            print("Iter: {:6d} | D_loss: {:8.4f} | G_loss: {:8.4f}".format(
                it + 1, D_loss_curr, G_loss_curr
            ))

        # 每 PLOT_EVERY iter 存一次模型並畫出生成樣本
        if (it + 1) % PLOT_EVERY == 0:
            ckpt_path = saver.save(
                sess,
                os.path.join(SAVE_MODEL_PATH,
                             "ChannelGAN_MIMO_step_{}.ckpt".format(it + 1))
            )
            print("Model saved to:", ckpt_path)

            plot_generated_samples(
                sess, G_sample, Z, Condition,
                h_dataset, SAVE_FIG_PATH, it + 1
            )

    # 5. 訓練結束後再存一次最終模型
    ckpt_path = saver.save(
        sess,
        os.path.join(SAVE_MODEL_PATH, "ChannelGAN_MIMO_final.ckpt")
    )
    print("Final model saved to:", ckpt_path)

    sess.close()


if __name__ == "__main__":
    main()
