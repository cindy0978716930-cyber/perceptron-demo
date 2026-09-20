import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HW1 - Perceptron Demo
#
# Part 1: Basic Case - AND Gate (class example)
# Part 2: Complex Case - Linearly Separable Data
# Part 3: Linear Unseparable Case - XOR
# ============================================================


# ------------------------------------------------------------
# Perceptron activation function
# hardlim(n):
#     n >= 0  -> 1
#     n < 0   -> 0
# ------------------------------------------------------------
def hardlim(n):
    if n >= 0:
        return 1
    else:
        return 0


# ------------------------------------------------------------
# Perceptron training function
#
# n = W dot p + b
# a = hardlim(n)
# e = t - a
#
# W(new) = W(old) + learning_rate * e * p
# b(new) = b(old) + learning_rate * e
# ------------------------------------------------------------
def train_perceptron(X, targets, initial_weights,
                     initial_bias=0, learning_rate=1,
                     max_epochs=20, show_steps=True):

    weights = np.array(initial_weights, dtype=float)
    bias = float(initial_bias)

    # Store information for plotting
    history = []
    errors_per_epoch = []

    print("\nInitial condition")
    print("W =", weights)
    print("b =", bias)
    print("Learning rate =", learning_rate)

    for epoch in range(1, max_epochs + 1):

        print("\n" + "=" * 65)
        print(f"Epoch {epoch}")
        print("=" * 65)

        error_count = 0

        for i in range(len(X)):

            p = X[i]
            t = targets[i]

            # Save old values
            old_weights = weights.copy()
            old_bias = bias

            # Step 1: calculate net input
            n = np.dot(weights, p) + bias

            # Step 2: activation function
            a = hardlim(n)

            # Step 3: calculate error
            e = t - a

            # Step 4: update weights and bias
            weights = weights + learning_rate * e * p
            bias = bias + learning_rate * e

            if e != 0:
                error_count += 1

            # Save history
            history.append({
                "epoch": epoch,
                "sample": i + 1,
                "input": p.copy(),
                "target": t,
                "old_weights": old_weights.copy(),
                "old_bias": old_bias,
                "net": n,
                "output": a,
                "error": e,
                "new_weights": weights.copy(),
                "new_bias": bias
            })

            # Show step-by-step process
            if show_steps:
                print(f"\nSample {i + 1}")
                print(f"Input p        = {p}")
                print(f"Target t       = {t}")
                print(f"Current W      = {old_weights}")
                print(f"Current b      = {old_bias:.2f}")

                print("\nForward:")
                print(f"n = W dot p + b = {n:.2f}")
                print(f"a = hardlim(n)  = {a}")

                print("\nError:")
                print(f"e = t - a       = {t} - {a} = {e}")

                if e == 0:
                    print("Correct -> no weight update")
                else:
                    print("\nUpdate:")
                    print(
                        f"W(new) = W(old) + eta * e * p"
                    )
                    print(
                        f"       = {old_weights} + "
                        f"{learning_rate} * {e} * {p}"
                    )
                    print(f"       = {weights}")

                    print(
                        f"b(new) = {old_bias:.2f} + "
                        f"{learning_rate} * {e}"
                    )
                    print(f"       = {bias:.2f}")

        errors_per_epoch.append(error_count)

        print("\n-----------------------------------")
        print(f"Epoch {epoch} finished")
        print(f"Number of errors = {error_count}")
        print(f"W = {weights}")
        print(f"b = {bias:.2f}")
        print("-----------------------------------")

        # If no errors, training has converged
        if error_count == 0:
            print(f"\n*** Converged at Epoch {epoch}! ***")
            return weights, bias, history, errors_per_epoch, True

    print("\n*** Did NOT converge within the maximum epochs. ***")

    return weights, bias, history, errors_per_epoch, False


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
def predict(X, weights, bias):

    predictions = []

    for p in X:
        n = np.dot(weights, p) + bias
        a = hardlim(n)
        predictions.append(a)

    return np.array(predictions)


# ------------------------------------------------------------
# Plot data and decision boundary
# ------------------------------------------------------------
def plot_decision_boundary(X, targets, weights, bias, title):

    plt.figure(figsize=(7, 6))

    # Class 0
    class0 = X[targets == 0]

    # Class 1
    class1 = X[targets == 1]

    plt.scatter(
        class0[:, 0],
        class0[:, 1],
        marker="o",
        s=100,
        label="Class 0"
    )

    plt.scatter(
        class1[:, 0],
        class1[:, 1],
        marker="x",
        s=100,
        label="Class 1"
    )

    # Decision boundary:
    #
    # w1*x1 + w2*x2 + b = 0
    #
    # x2 = -(w1*x1 + b) / w2
    #
    x_min = np.min(X[:, 0]) - 1
    x_max = np.max(X[:, 0]) + 1

    if abs(weights[1]) > 1e-10:

        x_values = np.linspace(x_min, x_max, 200)

        y_values = -(
            weights[0] * x_values + bias
        ) / weights[1]

        plt.plot(
            x_values,
            y_values,
            label="Decision Boundary"
        )

    elif abs(weights[0]) > 1e-10:

        x_boundary = -bias / weights[0]

        plt.axvline(
            x=x_boundary,
            label="Decision Boundary"
        )

    plt.xlabel("p1")
    plt.ylabel("p2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# ============================================================
# PART 1
# Basic Case: AND Gate
# ============================================================
def part1_and():

    print("\n")
    print("#" * 70)
    print("PART 1 - BASIC CASE: AND GATE")
    print("#" * 70)

    # AND truth table
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    targets = np.array([
        0,
        0,
        0,
        1
    ])

    # Same initial condition as the classroom example
    initial_weights = [1, 2]
    initial_bias = 0
    learning_rate = 1

    weights, bias, history, errors, converged = train_perceptron(
        X,
        targets,
        initial_weights=initial_weights,
        initial_bias=initial_bias,
        learning_rate=learning_rate,
        max_epochs=10,
        show_steps=True
    )

    predictions = predict(X, weights, bias)

    print("\n")
    print("=" * 50)
    print("PART 1 FINAL RESULT")
    print("=" * 50)

    print("Final W =", weights)
    print("Final b =", bias)

    print("\nPrediction:")
    print("p1 p2 | Target | Prediction")

    for p, t, a in zip(X, targets, predictions):
        print(
            f"{int(p[0])}  {int(p[1])}  |"
            f"   {t}    |     {a}"
        )

    print("\nExpected classroom result:")
    print("W = [1, 1]")
    print("b = -2")
    print("Decision boundary:")
    print("p1 + p2 - 2 = 0")

    plot_decision_boundary(
        X,
        targets,
        weights,
        bias,
        "Part 1: Perceptron Learning AND Gate"
    )


# ============================================================
# PART 2
# Complex Case: More linearly separable points
# ============================================================
def part2_complex():

    print("\n")
    print("#" * 70)
    print("PART 2 - COMPLEX LINEARLY SEPARABLE CASE")
    print("#" * 70)

    # More complex 2D data
    #
    # These two classes can still be separated by
    # one straight line.

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
        [0] * len(class0) +
        [1] * len(class1)
    )

    # Start from zero weights
    initial_weights = [0, 0]
    initial_bias = 0

    weights, bias, history, errors, converged = train_perceptron(
        X,
        targets,
        initial_weights=initial_weights,
        initial_bias=initial_bias,
        learning_rate=1,
        max_epochs=50,
        show_steps=False
    )

    predictions = predict(X, weights, bias)

    accuracy = np.mean(predictions == targets) * 100

    print("\n")
    print("=" * 50)
    print("PART 2 FINAL RESULT")
    print("=" * 50)

    print("Final W =", weights)
    print("Final b =", bias)
    print(f"Accuracy = {accuracy:.2f}%")

    if converged:
        print("Result: The data are linearly separable.")
        print("The perceptron successfully converged.")
    else:
        print("The perceptron did not converge.")

    print("\nErrors per epoch:")
    print(errors)

    plot_decision_boundary(
        X,
        targets,
        weights,
        bias,
        "Part 2: Complex Linearly Separable Case"
    )


# ============================================================
# PART 3
# Linear Unseparable Case: XOR
# ============================================================
def part3_xor():

    print("\n")
    print("#" * 70)
    print("PART 3 - LINEAR UNSEPARABLE CASE: XOR")
    print("#" * 70)

    # XOR truth table
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

    initial_weights = [0, 0]
    initial_bias = 0

    weights, bias, history, errors, converged = train_perceptron(
        X,
        targets,
        initial_weights=initial_weights,
        initial_bias=initial_bias,
        learning_rate=1,
        max_epochs=20,
        show_steps=False
    )

    predictions = predict(X, weights, bias)

    print("\n")
    print("=" * 50)
    print("PART 3 FINAL RESULT")
    print("=" * 50)

    print("Final W =", weights)
    print("Final b =", bias)

    print("\nXOR Prediction:")
    print("p1 p2 | Target | Prediction")

    for p, t, a in zip(X, targets, predictions):
        print(
            f"{int(p[0])}  {int(p[1])}  |"
            f"   {t}    |     {a}"
        )

    print("\nErrors per epoch:")
    print(errors)

    if not converged:
        print("\nResult:")
        print("The perceptron cannot converge for XOR.")
        print("XOR is NOT linearly separable.")
        print(
            "A single straight decision boundary "
            "cannot separate the two classes."
        )

    plot_decision_boundary(
        X,
        targets,
        weights,
        bias,
        "Part 3: XOR - Linearly Unseparable Case"
    )

    # Plot error count
    plt.figure(figsize=(7, 5))

    epochs = np.arange(1, len(errors) + 1)

    plt.plot(
        epochs,
        errors,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Number of Errors")
    plt.title(
        "XOR: Perceptron Does Not Converge"
    )

    plt.xticks(epochs)
    plt.grid(True)
    plt.tight_layout()


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():

    print("=" * 70)
    print("HW1 - PERCEPTRON DEMONSTRATION")
    print("=" * 70)

    print("""
This program demonstrates:

1. Basic Case
   AND gate with step-by-step calculation

2. Complex Case
   More linearly separable 2D data

3. Linear Unseparable Case
   XOR problem
""")

    # Run all three parts
    part1_and()
    part2_complex()
    part3_xor()

    # Show all figures
    plt.show()


if __name__ == "__main__":
    main()