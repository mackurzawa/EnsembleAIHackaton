import requests
from typing import List

def sybil_attack(ids: List[int], user: str, transformation: str):
    if user not in ["home", "defense"] or transformation not in ["binary", "affine"]:
        raise "Invalid endpoint"
    
    SERVER_URL="http://34.71.138.79:9090/"
    ENDPOINT = f"/sybil/{transformation}/{user}"
    URL = SERVER_URL + ENDPOINT
    
    TEAM_TOKEN = "Er6b7skOyWBCrtZC"
    ids = ids = ",".join(map(str, ids))

    response = requests.get(
        URL, params={"ids": ids}, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        return response.content["representations"]
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")
