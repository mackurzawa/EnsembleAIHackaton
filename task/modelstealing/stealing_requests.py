import requests

SERVER_URL="http://34.71.138.79:9090"
TEAM_TOKEN="Er6b7skOyWBCrtZC"

def model_stealing(path_to_png_file: str):
    endpoint = "/modelstealing"
    url = SERVER_URL + endpoint
    with open(path_to_png_file, "rb") as f:
        response = requests.get(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            representation = response.json()["representation"]

            return representation
        else:
            raise Exception(
                f"Model stealing failed. Code: {response.status_code}, content: {response.json()}"
            )


def model_stealing_submit(path_to_onnx_file: str):
    endpoint = "/modelstealing/submit"
    url = SERVER_URL + endpoint
    with open(path_to_onnx_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            print("Request ok")
            print(response.json())
        else:
            raise Exception(
                f"Model stealing submit failed. Code: {response.status_code}, content: {response.json()}"
            )


def model_stealing_reset():
    endpoint = f"/modelstealing/reset"
    url = SERVER_URL + endpoint
    response = requests.post(url, headers={"token": TEAM_TOKEN})
    if response.status_code == 200:
        print("Request ok")
        print(response.json())
    else:
        raise Exception(
            f"Model stealing reset failed. Code: {response.status_code}, content: {response.json()}"
        )
   

if __name__ == "__main__":
    print("💯🔥")
    print("🚀💻")
    print("😄🎉")
    # model_stealing_reset()
    # model_stealing("data/ModelStealingPub.png")
    # model_stealing_submission("modelstealing/models/example_submission.onnx")