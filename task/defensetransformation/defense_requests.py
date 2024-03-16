import requests
import numpy as np

SERVER_URL="http://34.71.138.79:9090"
TEAM_TOKEN="Er6b7skOyWBCrtZC"

def evaluate(eval_clf_performance: function, eval_mapper_performance: function, args, BASE_ACCURACY: float, ACCURACY_DROP_THRESHOLD: float):
    accuracy = eval_clf_performance(args)
    cosine_distance = eval_mapper_performance(args)

    accuracy_score = (
        0
        if accuracy < BASE_ACCURACY - ACCURACY_DROP_THRESHOLD
        else (
            1
            if accuracy > BASE_ACCURACY
            else (accuracy - BASE_ACCURACY + ACCURACY_DROP_THRESHOLD)
            / ACCURACY_DROP_THRESHOLD
        )
    )
    return accuracy_score + cosine_distance

def is_my_data_float32_uwu(path: str):
    data = np.load([path], allow_pickle=True)
    print(data["representations"].dtype)

# Be careful. This can be done only once an hour.
# Computing this might take a few minutes. Be patient.
# Make sure your file has proper content.
def defense_submit(path_to_npz_file: str):
    endpoint = "/defense/submit"
    url = SERVER_URL + endpoint
    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            print("Request ok")
            print(response.json())
        else:
            raise Exception(
                f"Defense submit failed. Code: {response.status_code}, content: {response.json()}"
            )
        

if __name__ == "__main__":
    print("You can do it!")
    print("Believe in yourself and keep pushing forward!")
    print("Success is just around the corner!")
    # defense_submit("data/ExampleDefenseTransformationSubmit.npz")