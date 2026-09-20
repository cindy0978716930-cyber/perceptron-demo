import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="感知器互動學習展示",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 感知器（Perceptron）互動學習展示")
st.write(
    "HW1 — 基礎案例、進階案例與 "
    "XOR 線性不可分案例"
)

# ============================================================
# Basic functions
# ============================================================

def hardlim(n):
    """Hard limit activation function."""
    return 1 if n >= 0 else 0


def one_epoch(X, targets, weights, bias, learning_rate=1):
    """
    Train the perceptron for one epoch.
    Return updated weights, bias, and step-by-step records.
    """
    weights = weights.copy()
    records = []
    error_count = 0

    for i in range(len(X)):
        p = X[i]
        t = targets[i]

        old_w = weights.copy()
        old_b = bias

        # Forward
        n = np.dot(weights, p) + bias
        a = hardlim(n)

        # Error
        e = t - a

        # Update
        weights = weights + learning_rate * e * p
        bias = bias + learning_rate * e

        if e != 0:
            error_count += 1

        records.append({
            "Sample": i + 1,
            "Input p": str(tuple(p.astype(int))),
            "Target t": int(t),
            "Old W": str(old_w),
            "Old b": old_b,
            "n = W·p+b": n,
            "Output a": a,
            "Error e": e,
            "New W": str(weights),
            "New b": bias
        })

    return weights, bias, records, error_count


def train_until_end(
    X,
    targets,
    initial_weights,
    initial_bias,
    learning_rate=1,
    max_epochs=20
):
    """Train until convergence or maximum number of epochs."""
    weights = np.array(initial_weights, dtype=float)
    bias = float(initial_bias)

    all_records = []
    errors = []
    boundary_history = []

    for epoch in range(1, max_epochs + 1):

        weights, bias, records, error_count = one_epoch(
            X,
            targets,
            weights,
            bias,
            learning_rate
        )

        for record in records:
            record["Epoch"] = epoch

        all_records.extend(records)
        errors.append(error_count)
        boundary_history.append((weights.copy(), bias))

        if error_count == 0:
            return (
                weights,
                bias,
                all_records,
                errors,
                True,
                epoch,
                boundary_history
            )

    return (
        weights,
        bias,
        all_records,
        errors,
        False,
        max_epochs,
        boundary_history
    )


def draw_boundary(X, targets, weights, bias, title):
    """Draw data points and current decision boundary."""

    fig, ax = plt.subplots(figsize=(6, 5))

    class0 = X[targets == 0]
    class1 = X[targets == 1]

    ax.scatter(
        class0[:, 0],
        class0[:, 1],
        s=100,
        marker="o",
        label="Class 0"
    )

    ax.scatter(
        class1[:, 0],
        class1[:, 1],
        s=100,
        marker="x",
        label="Class 1"
    )

    margin = 1
    xmin = np.min(X[:, 0]) - margin
    xmax = np.max(X[:, 0]) + margin
    ymin = np.min(X[:, 1]) - margin
    ymax = np.max(X[:, 1]) + margin

    # W1*x1 + W2*x2 + b = 0
    if abs(weights[1]) > 1e-10:

        x_values = np.linspace(xmin, xmax, 200)

        y_values = -(
            weights[0] * x_values + bias
        ) / weights[1]

        ax.plot(
            x_values,
            y_values,
            linewidth=2,
            label="Decision Boundary"
        )

    elif abs(weights[0]) > 1e-10:

        x_boundary = -bias / weights[0]

        ax.axvline(
            x_boundary,
            linewidth=2,
            label="Decision Boundary"
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_title(title)

    ax.grid(True)
    ax.legend()

    return fig


def animate_training(
    X,
    targets,
    initial_weights,
    initial_bias,
    learning_rate=1,
    max_epochs=20,
    delay=0.35,
    title_prefix="Training"
):
    """
    Visualize the perceptron learning process sample by sample.
    The decision boundary moves only according to the actual
    perceptron updates produced by one_epoch().
    """
    weights = np.array(initial_weights, dtype=float)
    bias = float(initial_bias)

    plot_box = st.empty()
    status_box = st.empty()
    formula_box = st.empty()
    progress_box = st.empty()

    total_steps = max_epochs * len(X)
    completed_steps = 0
    errors_history = []
    converged = False

    for epoch in range(1, max_epochs + 1):
        weights, bias, records, error_count = one_epoch(
            X, targets, weights, bias, learning_rate
        )
        errors_history.append(error_count)

        # one_epoch() record keys are internal data keys; keep them language-independent.
        for record in records:
            completed_steps += 1

            with status_box.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Epoch", epoch)
                c2.metric("Sample", record["Sample"])
                c3.metric("誤差 e", int(record["Error e"]))
                c4.metric(
                    "動作",
                    "更新" if record["Error e"] != 0 else "不更新"
                )

            with formula_box.container():
                st.caption(
                    f"p = {record['Input p']}   |   "
                    f"t = {record['Target t']}   |   "
                    f"n = {record['n = W·p+b']:.2f}   |   "
                    f"a = {record['Output a']}"
                )
                st.write(
                    f"**W:** {record['Old W']} → {record['New W']}    "
                    f"**b:** {record['Old b']:.2f} → {record['New b']:.2f}"
                )

            fig = draw_boundary(
                X,
                targets,
                np.array(
                    record["New W"].strip("[]").split(),
                    dtype=float
                ),
                float(record["New b"]),
                f"{title_prefix} — Epoch {epoch}, Sample {record['Sample']}"
            )
            plot_box.pyplot(fig)
            plt.close(fig)

            progress_box.progress(
                min(completed_steps / total_steps, 1.0),
                text=f"Epoch {epoch}: {error_count} error(s) in this epoch"
            )
            time.sleep(delay)

        if error_count == 0:
            converged = True
            break

    return weights, bias, errors_history, converged, epoch


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("控制面板")

mode = st.sidebar.selectbox(
    "選擇展示模式",
    [
        "1. 基礎案例 — AND 邏輯閘",
        "2. 進階案例 — 線性可分",
        "3. 限制案例 — XOR 線性不可分"
    ]
)

st.sidebar.divider()

st.sidebar.write("### 感知器學習公式")

st.sidebar.latex(r"n = Wp + b")
st.sidebar.latex(r"a = hardlim(n)")
st.sidebar.latex(r"e = t-a")
st.sidebar.latex(r"W_{new}=W_{old}+\eta ep^T")
st.sidebar.latex(r"b_{new}=b_{old}+\eta e")


# ============================================================
# PART 1 — AND
# ============================================================

if mode == "1. 基礎案例 — AND 邏輯閘":

    st.header("Part 1｜基礎案例：AND 邏輯閘")

    st.info(
        "本區重現老師課堂中的 AND 範例，"
        "並逐步展示感知器的計算與權重更新。"
    )

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    targets = np.array([0, 0, 0, 1])

    st.subheader("訓練資料（Training Data）")

    st.table({
        "p1": [0, 0, 1, 1],
        "p2": [0, 1, 0, 1],
        "目標值 t": [0, 0, 0, 1]
    })

    st.write("### 初始設定（Initial Conditions）")

    col1, col2, col3 = st.columns(3)

    col1.metric("初始權重 W", "[1, 2]")
    col2.metric("初始偏置 b", "0")
    col3.metric("學習率 η", "1")

    st.divider()

    # ------------------------------------------------------------
    # Classroom overview: always show the three key AND boundaries
    # Initial -> Correction 1 -> Correction 2 (final)
    # ------------------------------------------------------------
    st.subheader("AND 學習過程總覽")
    st.caption(
        "固定呈現三個關鍵決策邊界：初始狀態 → 第 1 次修正 → 第 2 次修正。"
    )

    overview1, overview2, overview3 = st.columns(3)

    with overview1:
        fig0 = draw_boundary(
            X,
            targets,
            np.array([1.0, 2.0]),
            0.0,
            "Initial Boundary  W=[1,2], b=0"
        )
        st.pyplot(fig0, use_container_width=True)
        plt.close(fig0)
        st.latex(r"p_1+2p_2=0")
        st.caption("尚未進行任何修正")

    with overview2:
        fig1 = draw_boundary(
            X,
            targets,
            np.array([1.0, 2.0]),
            -1.0,
            "Correction 1  W=[1,2], b=-1"
        )
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
        st.latex(r"p_1+2p_2-1=0")
        st.caption("第一次分類錯誤後更新")

    with overview3:
        fig2 = draw_boundary(
            X,
            targets,
            np.array([1.0, 1.0]),
            -2.0,
            "Correction 2 / Final  W=[1,1], b=-2"
        )
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.latex(r"p_1+p_2-2=0")
        st.caption("最終決策邊界；下一個完整週期零錯誤後確認收斂")

    st.info(
        "學習流程：初始決策邊界 → 第 1 次修正 → "
        "第 2 次修正 → 驗證完整零錯誤週期 → 收斂。"
    )

    st.divider()

    if "and_step" not in st.session_state:
        st.session_state.and_step = 0

    # Pre-calculate all steps
    weights = np.array([1.0, 2.0])
    bias = 0.0
    steps = []

    for epoch in range(1, 10):

        errors = 0

        for i in range(len(X)):

            p = X[i]
            t = targets[i]

            old_w = weights.copy()
            old_b = bias

            n = np.dot(weights, p) + bias
            a = hardlim(n)
            e = t - a

            weights = weights + e * p
            bias = bias + e

            if e != 0:
                errors += 1

            steps.append({
                "epoch": epoch,
                "sample": i + 1,
                "p": p.copy(),
                "t": t,
                "old_w": old_w,
                "old_b": old_b,
                "n": n,
                "a": a,
                "e": e,
                "new_w": weights.copy(),
                "new_b": bias
            })

        if errors == 0:
            break

    st.subheader("逐步學習過程（Step-by-Step）")

    c1, c2, c3 = st.columns(3)

    if c1.button("⬅ 上一步"):
        if st.session_state.and_step > 0:
            st.session_state.and_step -= 1

    if c2.button("➡ 下一步"):
        if st.session_state.and_step < len(steps) - 1:
            st.session_state.and_step += 1

    if c3.button("🔄 重設"):
        st.session_state.and_step = 0

    st.caption("手動模式可逐筆檢查資料；自動播放會展示決策邊界如何隨學習過程移動。")
    speed_label = st.select_slider(
        "動畫速度",
        options=["慢速", "正常", "快速"],
        value="正常",
        key="and_speed"
    )
    speed_map = {"慢速": 0.8, "正常": 0.4, "快速": 0.15}

    if st.button("▶ 自動播放 AND 學習過程", key="and_autoplay", use_container_width=True):
        animation_plot = st.empty()
        animation_info = st.empty()
        animation_progress = st.empty()

        # Frame 0: show the classroom INITIAL boundary before any sample is processed.
        initial_w = np.array([1.0, 2.0])
        initial_b = 0.0
        with animation_info.container():
            st.markdown("### 畫面 0 — 初始狀態")
            st.write("**W = [1, 2], b = 0** · 尚未進行任何修正。")
            st.latex(r"p_1 + 2p_2 = 0")
        fig_anim = draw_boundary(
            X, targets, initial_w, initial_b,
            "AND — Initial Boundary  W=[1,2], b=0"
        )
        animation_plot.pyplot(fig_anim)
        plt.close(fig_anim)
        animation_progress.progress(0.0, text="初始狀態")
        time.sleep(speed_map[speed_label])

        # For the AND classroom animation, emphasize actual CORRECTIONS.
        # This gives the same conceptual sequence as the lecture:
        # initial -> correction 1 -> correction 2 -> convergence.
        correction_no = 0
        total_corrections = sum(1 for s in steps if s["e"] != 0)

        for j, animation_step in enumerate(steps):
            if animation_step["e"] == 0:
                continue

            correction_no += 1
            with animation_info.container():
                st.markdown(
                    f"### Correction {correction_no} — "
                    f"Epoch {animation_step['epoch']}, Sample {animation_step['sample']}"
                )
                aa, bb, cc, dd = st.columns(4)
                aa.metric(
                    "輸入 p",
                    f"({int(animation_step['p'][0])}, {int(animation_step['p'][1])})"
                )
                bb.metric("目標值 t", int(animation_step["t"]))
                cc.metric("輸出值 a", int(animation_step["a"]))
                dd.metric("誤差 e", int(animation_step["e"]))

                st.latex(
                    rf"n=W\cdot p+b={animation_step['n']:.1f},\quad "
                    rf"a=hardlim(n)={int(animation_step['a'])},\quad "
                    rf"e=t-a={int(animation_step['e'])}"
                )
                st.write(
                    f"**W:** {animation_step['old_w']} → {animation_step['new_w']}    "
                    f"**b:** {animation_step['old_b']:.1f} → {animation_step['new_b']:.1f}"
                )

            fig_anim = draw_boundary(
                X,
                targets,
                animation_step["new_w"],
                animation_step["new_b"],
                f"AND — Correction {correction_no}/{total_corrections}"
            )
            animation_plot.pyplot(fig_anim)
            plt.close(fig_anim)
            animation_progress.progress(
                correction_no / total_corrections,
                text=f"Correction {correction_no} of {total_corrections}"
            )
            time.sleep(speed_map[speed_label])

        # Hold the final converged boundary and explain why training stops.
        with animation_info.container():
            st.success("✅ Converged after the next full epoch contains zero errors.")
            st.write("**Final W = [1, 1], b = -2**")
            st.latex(r"p_1+p_2-2=0")
            st.caption(
                "Classroom sequence: Initial boundary → Correction 1 → "
                "Correction 2 → verify one zero-error epoch → Converged."
            )

        st.session_state.and_step = len(steps) - 1

    step = steps[st.session_state.and_step]

    st.progress(
        (st.session_state.and_step + 1) / len(steps)
    )

    st.write(
        f"### Epoch {step['epoch']} — "
        f"Sample {step['sample']}"
    )

    left, right = st.columns(2)

    with left:

        st.write("#### 目前資料（Current Sample）")

        st.write(
            f"**輸入：** p = "
            f"({int(step['p'][0])}, "
            f"{int(step['p'][1])})"
        )

        st.write(f"**目標值：** t = {step['t']}")

        st.write(
            f"**目前 W：** {step['old_w']}"
        )

        st.write(
            f"**目前 b：** {step['old_b']}"
        )

        st.write("#### ① 前向計算（Forward）")

        st.latex(
            rf"n = Wp+b = {step['n']:.1f}"
        )

        st.latex(
            rf"a = hardlim(n) = {step['a']}"
        )

        st.write("#### ② 誤差（Error）")

        st.latex(
            rf"e=t-a={step['t']}-{step['a']}"
            rf"={step['e']}"
        )

        st.write("#### ③ 權重更新（Update）")

        if step["e"] == 0:
            st.success(
                "預測正確 → "
                "權重與偏置不需更新。"
            )
        else:
            st.latex(
                r"W_{new}=W_{old}+\eta ep^T"
            )

            st.write(
                "**更新後 W：**",
                step["new_w"]
            )

            st.write(
                "**更新後 b：**",
                step["new_b"]
            )

    with right:

        fig = draw_boundary(
            X,
            targets,
            step["new_w"],
            step["new_b"],
            "AND Decision Boundary"
        )

        st.pyplot(fig)

    if st.session_state.and_step == len(steps) - 1:

        st.success(
            "訓練已收斂！"
            "Final W = [1, 1], b = -2."
        )

        st.latex(
            r"p_1+p_2-2=0"
        )


# ============================================================
# PART 2 — COMPLEX
# ============================================================

elif mode == "2. 進階案例 — 線性可分":

    st.header(
        "Part 2｜進階案例：線性可分資料"
    )

    st.write(
        "此資料集比 AND 範例包含更多資料點，"
        "但兩個類別仍然可以"
        "由一條直線完全分開。"
    )

    class0 = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.5, 1.5],
        [2.0, 1.0],
        [2.0, 2.0],
        [2.5, 1.0],
        [1.0, 2.5],
        [2.3, 1.7],
        [1.4, 2.3],
        [2.6, 1.5]
    ])

    class1 = np.array([
        [4.0, 4.0],
        [4.0, 5.0],
        [4.5, 4.5],
        [5.0, 4.0],
        [5.0, 5.0],
        [5.5, 4.0],
        [4.0, 5.5],
        [4.7, 5.2],
        [5.3, 4.6],
        [3.8, 4.7]
    ])

    X = np.vstack((class0, class1))

    targets = np.array(
        [0] * len(class0)
        + [1] * len(class1)
    )

    initial_weights = [0, 0]
    initial_bias = 0

    before_fig = draw_boundary(
        X,
        targets,
        np.array(initial_weights, dtype=float),
        initial_bias,
        "Before Training"
    )

    st.pyplot(before_fig)

    st.write("### 觀察學習過程")
    st.caption(
        "移動中的直線代表目前的決策邊界。"
        "當資料被錯誤分類時，W 與 b 會更新，決策邊界也會隨之移動。"
    )
    complex_speed = st.select_slider(
        "進階案例動畫速度",
        options=["慢速", "正常", "快速"],
        value="正常",
        key="complex_speed"
    )
    complex_speed_map = {"慢速": 0.65, "正常": 0.30, "快速": 0.10}

    if st.button("🎬 播放訓練動畫", key="complex_animation", use_container_width=True):
        aw, ab, aerrors, aconverged, aepoch = animate_training(
            X,
            targets,
            initial_weights,
            initial_bias,
            learning_rate=1,
            max_epochs=50,
            delay=complex_speed_map[complex_speed],
            title_prefix="Complex Case"
        )
        if aconverged:
            st.success(
                f"✅ Converged after {aepoch} epoch(s). "
                "The final epoch has zero classification errors."
            )

    if st.button("▶ 訓練感知器"):

        (
            weights,
            bias,
            records,
            errors,
            converged,
            epoch,
            history
        ) = train_until_end(
            X,
            targets,
            initial_weights,
            initial_bias,
            learning_rate=1,
            max_epochs=50
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "是否收斂",
            "Yes" if converged else "No"
        )

        col2.metric(
            "訓練週期（Epochs）",
            epoch
        )

        predictions = np.array([
            hardlim(np.dot(weights, p) + bias)
            for p in X
        ])

        accuracy = np.mean(
            predictions == targets
        ) * 100

        col3.metric(
            "準確率（Accuracy）",
            f"{accuracy:.1f}%"
        )

        st.write("### 最終參數")

        st.write("**W =**", weights)
        st.write("**b =**", bias)

        final_fig = draw_boundary(
            X,
            targets,
            weights,
            bias,
            "After Training — Learned Decision Boundary"
        )

        st.pyplot(final_fig)

        st.success(
            "感知器成功收斂，因為"
            "此資料集為線性可分。"
        )

        st.write("### 每個 Epoch 的錯誤數")

        fig2, ax2 = plt.subplots(figsize=(7, 4))

        ax2.plot(
            range(1, len(errors) + 1),
            errors,
            marker="o"
        )

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Number of Errors")
        ax2.set_title("Training Error")
        ax2.grid(True)

        st.pyplot(fig2)


# ============================================================
# PART 3 — XOR
# ============================================================

else:

    st.header(
        "Part 3｜限制案例：XOR 線性不可分"
    )

    st.warning(
        "XOR 無法使用"
        "單一線性決策邊界完全分開。"
    )

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    targets = np.array([
        0,
        1,
        1,
        0
    ])

    st.subheader("XOR Data")

    st.table({
        "p1": [0, 0, 1, 1],
        "p2": [0, 1, 0, 1],
        "目標值 t": [0, 1, 1, 0]
    })

    initial_weights = [0, 0]
    initial_bias = 0

    original_fig = draw_boundary(
        X,
        targets,
        np.array(initial_weights, dtype=float),
        initial_bias,
        "XOR Data"
    )

    st.pyplot(original_fig)

    max_epochs = st.slider(
        "最大訓練週期（Maximum Epochs）",
        min_value=5,
        max_value=50,
        value=20
    )

    st.write("### 觀察 XOR 為何無法收斂")
    st.caption(
        "訓練過程中，決策邊界會因不同的錯誤分類資料而持續被修正。"
        "由於 XOR 並非線性可分，單一條直線無法同時正確分類四個資料點。"
    )
    xor_speed = st.select_slider(
        "XOR 動畫速度",
        options=["慢速", "正常", "快速"],
        value="正常",
        key="xor_speed"
    )
    xor_speed_map = {"慢速": 0.65, "正常": 0.30, "快速": 0.10}

    if st.button("🎬 播放 XOR 訓練動畫", key="xor_animation", use_container_width=True):
        xw, xb, xerrors, xconverged, xepoch = animate_training(
            X,
            targets,
            initial_weights,
            initial_bias,
            learning_rate=1,
            max_epochs=max_epochs,
            delay=xor_speed_map[xor_speed],
            title_prefix="XOR"
        )
        if not xconverged:
            st.error(
                f"❌ No convergence after {xepoch} epoch(s). "
                "The moving boundary demonstrates the limitation of a single perceptron on XOR."
            )

    if st.button("▶ 嘗試訓練 XOR"):

        (
            weights,
            bias,
            records,
            errors,
            converged,
            epoch,
            history
        ) = train_until_end(
            X,
            targets,
            initial_weights,
            initial_bias,
            learning_rate=1,
            max_epochs=max_epochs
        )

        st.divider()

        col1, col2 = st.columns(2)

        col1.metric(
            "是否收斂",
            "Yes" if converged else "No"
        )

        col2.metric(
            "已嘗試 Epoch 數",
            epoch
        )

        final_fig = draw_boundary(
            X,
            targets,
            weights,
            bias,
            "XOR — Final Decision Boundary"
        )

        st.pyplot(final_fig)

        st.write("### 每個 Epoch 的錯誤數")

        fig2, ax2 = plt.subplots(figsize=(7, 4))

        ax2.plot(
            range(1, len(errors) + 1),
            errors,
            marker="o"
        )

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Number of Errors")
        ax2.set_title(
            "XOR Does Not Reach Zero Error"
        )

        ax2.grid(True)

        st.pyplot(fig2)

        if not converged:

            st.error(
                "感知器未能收斂。"
            )

            st.write(
                """
                **為什麼？**

                單層感知器（Single-layer Perceptron）只能建立
                **線性決策邊界（Linear Decision Boundary）**。

                然而 XOR **不是線性可分（Linearly Separable）**。
                不存在任何一條直線，可以同時正確分開四個 XOR 資料點。

                因此，單一感知器無法完美解決 XOR 問題。
                """
            )
            