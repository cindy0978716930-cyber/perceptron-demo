import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
            