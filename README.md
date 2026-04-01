def generate_real_samples_with_labels_Rayleigh(h_dataset, number=100):
    """
    Generate real (labeled) samples for training the CGAN.
    
    Steps:
        1. 隨機從 h_dataset 抽取通道係數 h。
        2. 為每個樣本隨機產生 16QAM symbol。
        3. 模擬通道輸出 y = h*x + n，其中 n 為高斯雜訊。
        4. 建立 conditioning 向量 [Re(x), Im(x), Re(h), Im(h)] / 3。

    Args:
        h_dataset: 1-D array-like, 含有複數通道係數 h_siso。
        number: 要產生的樣本數。

    Returns:
        received_data: shape = (number, 2)，每列為 [Re(y), Im(y)]。
        conditioning:  shape = (number, 4)，每列為 [Re(x), Im(x), Re(h), Im(h)] / 3。
    """
    # 1. 隨機選 h（複數 Rayleigh 通道係數）
    h_complex = np.random.choice(h_dataset, size=number, replace=True)  # shape: (number,)
    h_r = np.real(h_complex)
    h_i = np.imag(h_complex)

    # 2. 產生隨機 16QAM symbol（使用全域的 mean_set_QAM）
    #    隨機挑 index，然後從 mean_set_QAM 取對應 symbol
    symbol_idx = np.random.randint(low=0, high=len(mean_set_QAM), size=number)
    x_complex = mean_set_QAM[symbol_idx]  # shape: (number,)
    x_r = np.real(x_complex)
    x_i = np.imag(x_complex)

    # 3. 通道輸出 y = h * x + n
    #    先計算 h*x
    y_complex = h_complex * x_complex  # shape: (number,)

    #    加上高斯雜訊（和你繪圖區使用的雜訊模型一致）
    noise = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[0.03, 0], [0, 0.03]],
        size=number
    ).astype(np.float32)
    # y 的實部、虛部
    y_r = np.real(y_complex)
    y_i = np.imag(y_complex)

    #    把雜訊加進去
    y_r_noisy = y_r + noise[:, 0]
    y_i_noisy = y_i + noise[:, 1]

    # 4. 建構輸出與 conditioning
    # received_data: [Re(y), Im(y)]
    received_data = np.stack([y_r_noisy, y_i_noisy], axis=1).astype(np.float32)

    # conditioning: [Re(x), Im(x), Re(h), Im(h)] / 3
    conditioning = np.stack([x_r, x_i, h_r, h_i], axis=1).astype(np.float32) / 3.0

    return received_data, conditioning
