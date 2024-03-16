import requests

def sybil_attack_reset():
    SERVER_URL="http://34.71.138.79:9090/"
    ENDPOINT = "/sybil/reset"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "Er6b7skOyWBCrtZC"

    response = requests.post(
        URL, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        print("Endpoint rested successfully")
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")