import numpy as np

if __name__ == "__main__":
    data = np.load(
        "data/ExampleDefenseTransformationEvaluate.npz"
    )
    print(data["labels"], data["representations"].shape)

    data = np.load("data/ExampleDefenseTransformationSubmit.npz")
    print(data["representations"].shape)
