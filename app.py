import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="Perceptron Demo",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Perceptron Demonstration")
st.write(
    "HW1 — Basic Case, Complex Case, "
    "and Linear Unseparable Case"
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

        for record in records:
            completed_steps += 1

            with status_box.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Epoch", epoch)
                c2.metric("Sample", record["Sample"])
                c3.metric("Error e", int(record["Error e"]))
                c4.metric(
                    "Action",
                    "Update" if record["Error e"] != 0 else "No change"
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

st.sidebar.header("Control Panel")

mode = st.sidebar.selectbox(
    "Choose a demonstration mode",
    [
        "1. Basic Case — AND Gate",
        "2. Complex Case — Linearly Separable",
        "3. Linear Unseparable Case — XOR"
    ]
)

st.sidebar.divider()

st.sidebar.write("### Perceptron Formula")

st.sidebar.latex(r"n = Wp + b")
st.sidebar.latex(r"a = hardlim(n)")
st.sidebar.latex(r"e = t-a")
st.sidebar.latex(r"W_{new}=W_{old}+\eta ep^T")
st.sidebar.latex(r"b_{new}=b_{old}+\eta e")


# ============================================================
# PART 1 — AND
# ============================================================

if mode == "1. Basic Case — AND Gate":

    st.header("Part 1 — Basic Case: AND Gate")

    st.info(
        "This mode reproduces the classroom example "
        "step by step."
    )

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    targets = np.array([0, 0, 0, 1])

    st.subheader("Training Data")

    st.table({
        "p1": [0, 0, 1, 1],
        "p2": [0, 1, 0, 1],
        "Target t": [0, 0, 0, 1]
    })

    st.write("### Initial Conditions")

    col1, col2, col3 = st.columns(3)

    col1.metric("Initial W", "[1, 2]")
    col2.metric("Initial b", "0")
    col3.metric("Learning Rate η", "1")

    st.divider()

    # ------------------------------------------------------------
    # Classroom overview: always show the three key AND boundaries
    # Initial -> Correction 1 -> Correction 2 (final)
    # ------------------------------------------------------------
    st.subheader("AND Learning Process Overview")
    st.caption(
        "These three figures stay on screen so the initial boundary is never skipped."
    )

    overview1, overview2, overview3 = st.columns(3)

    with overview1:
        fig0 = draw_boundary(
            X,
            targets,
            np.array([1.0, 2.0]),
            0.0,
            "Initial  W=[1,2], b=0"
        )
        st.pyplot(fig0, use_container_width=True)
        plt.close(fig0)
        st.latex(r"p_1+2p_2=0")
        st.caption("Before any correction")

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
        st.caption("After the first misclassification update")

    with overview3:
        fig2 = draw_boundary(
            X,
            targets,
            np.array([1.0, 1.0]),
            -2.0,
            "Correction 2  W=[1,1], b=-2"
        )
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.latex(r"p_1+p_2-2=0")
        st.caption("Final boundary; next zero-error epoch confirms convergence")

    st.info(
        "Learning sequence: Initial boundary → Correction 1 → "
        "Correction 2 → verify one complete zero-error epoch → Converged."
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

    st.subheader("Step-by-Step Learning")

    c1, c2, c3 = st.columns(3)

    if c1.button("⬅ Previous Step"):
        if st.session_state.and_step > 0:
            st.session_state.and_step -= 1

    if c2.button("➡ Next Step"):
        if st.session_state.and_step < len(steps) - 1:
            st.session_state.and_step += 1

    if c3.button("🔄 Reset"):
        st.session_state.and_step = 0

    st.caption("Manual mode lets you inspect each sample. Auto Play shows the decision boundary evolving automatically.")
    speed_label = st.select_slider(
        "Animation speed",
        options=["Slow", "Normal", "Fast"],
        value="Normal",
        key="and_speed"
    )
    speed_map = {"Slow": 0.8, "Normal": 0.4, "Fast": 0.15}

    if st.button("▶ Auto Play AND Learning", key="and_autoplay", use_container_width=True):
        animation_plot = st.empty()
        animation_info = st.empty()
        animation_progress = st.empty()

        # Frame 0: show the classroom INITIAL boundary before any sample is processed.
        initial_w = np.array([1.0, 2.0])
        initial_b = 0.0
        with animation_info.container():
            st.markdown("### Frame 0 — Initial condition")
            st.write("**W = [1, 2], b = 0** · No correction has been made yet.")
            st.latex(r"p_1 + 2p_2 = 0")
        fig_anim = draw_boundary(
            X, targets, initial_w, initial_b,
            "AND — Initial Boundary  W=[1,2], b=0"
        )
        animation_plot.pyplot(fig_anim)
        plt.close(fig_anim)
        animation_progress.progress(0.0, text="Initial state")
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
                    "Input p",
                    f"({int(animation_step['p'][0])}, {int(animation_step['p'][1])})"
                )
                bb.metric("Target t", int(animation_step["t"]))
                cc.metric("Output a", int(animation_step["a"]))
                dd.metric("Error e", int(animation_step["e"]))

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

        st.write("#### Current Sample")

        st.write(
            f"**Input:** p = "
            f"({int(step['p'][0])}, "
            f"{int(step['p'][1])})"
        )

        st.write(f"**Target:** t = {step['t']}")

        st.write(
            f"**Current W:** {step['old_w']}"
        )

        st.write(
            f"**Current b:** {step['old_b']}"
        )

        st.write("#### ① Forward")

        st.latex(
            rf"n = Wp+b = {step['n']:.1f}"
        )

        st.latex(
            rf"a = hardlim(n) = {step['a']}"
        )

        st.write("#### ② Error")

        st.latex(
            rf"e=t-a={step['t']}-{step['a']}"
            rf"={step['e']}"
        )

        st.write("#### ③ Update")

        if step["e"] == 0:
            st.success(
                "Correct prediction → "
                "weights do not change."
            )
        else:
            st.latex(
                r"W_{new}=W_{old}+\eta ep^T"
            )

            st.write(
                "**New W:**",
                step["new_w"]
            )

            st.write(
                "**New b:**",
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
            "Training converged! "
            "Final W = [1, 1], b = -2."
        )

        st.latex(
            r"p_1+p_2-2=0"
        )


# ============================================================
# PART 2 — COMPLEX
# ============================================================

elif mode == "2. Complex Case — Linearly Separable":

    st.header(
        "Part 2 — Complex Linearly Separable Case"
    )

    st.write(
        "The dataset contains more points than the AND "
        "example, but the two classes can still be "
        "separated by one straight line."
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

    st.write("### Watch the Learning Process")
    st.caption(
        "The moving line is the current decision boundary. "
        "When a sample is misclassified, W and b are updated and the line moves."
    )
    complex_speed = st.select_slider(
        "Complex animation speed",
        options=["Slow", "Normal", "Fast"],
        value="Normal",
        key="complex_speed"
    )
    complex_speed_map = {"Slow": 0.65, "Normal": 0.30, "Fast": 0.10}

    if st.button("🎬 Play Training Animation", key="complex_animation", use_container_width=True):
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

    if st.button("▶ Train Perceptron"):

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
            "Converged",
            "Yes" if converged else "No"
        )

        col2.metric(
            "Epochs",
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
            "Accuracy",
            f"{accuracy:.1f}%"
        )

        st.write("### Final Parameters")

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
            "The perceptron converged because "
            "the data are linearly separable."
        )

        st.write("### Errors per Epoch")

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
        "Part 3 — Linear Unseparable Case: XOR"
    )

    st.warning(
        "XOR cannot be perfectly separated by "
        "one straight decision boundary."
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
        "Target t": [0, 1, 1, 0]
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
        "Maximum Epochs",
        min_value=5,
        max_value=50,
        value=20
    )

    st.write("### Watch Why XOR Fails")
    st.caption(
        "During training, the boundary keeps being corrected for different samples. "
        "Because XOR is not linearly separable, a single straight line cannot satisfy all four points."
    )
    xor_speed = st.select_slider(
        "XOR animation speed",
        options=["Slow", "Normal", "Fast"],
        value="Normal",
        key="xor_speed"
    )
    xor_speed_map = {"Slow": 0.65, "Normal": 0.30, "Fast": 0.10}

    if st.button("🎬 Play XOR Training Animation", key="xor_animation", use_container_width=True):
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

    if st.button("▶ Try to Train XOR"):

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
            "Converged",
            "Yes" if converged else "No"
        )

        col2.metric(
            "Epochs Tried",
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

        st.write("### Errors per Epoch")

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
                "The perceptron did not converge."
            )

            st.write(
                """
                **Why?**

                A single-layer perceptron creates a
                linear decision boundary.

                However, XOR is **not linearly
                separable**. There is no single
                straight line that can correctly
                separate all four XOR points.

                Therefore, a single perceptron
                cannot solve XOR perfectly.
                """
            )
            