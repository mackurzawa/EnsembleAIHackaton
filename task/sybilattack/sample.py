import numpy as np
import pandas as pd
from sybil_attack import sybil_attack
from sybil_reset import sybil_attack_reset

NUMBER_OF_EXAMPLES = 20000
NUMBER_OF_COMMON_EXAMPLES = 615
NUMBER_OF_CLUSTERS = 14

# users
USERS = ["home", "defense"]
TRANSFORMATIONS = ["binary", "affine"]


def dummy_embeds(ids: list[int], user: str, transform: str):
    return np.random.rand(len(ids), 192)

# generate sample data
example_data = np.zeros(NUMBER_OF_EXAMPLES)

for i in range(NUMBER_OF_COMMON_EXAMPLES, NUMBER_OF_EXAMPLES):
    example_data[i] = i%14 + 1

labels, counts = np.unique(example_data, return_counts=True)
assert np.sum(counts) == NUMBER_OF_EXAMPLES

# 
common_img_ids = np.where(example_data == 0)[0]


df_common_embeds = pd.DataFrame(columns=["img_id", "embedding", "transformation"])
df_other_embeds = pd.DataFrame(columns=["img_id", "embedding", "transformation"])

for i in range(1, NUMBER_OF_CLUSTERS+1):
    img_ids = np.concatenate([common_img_ids, np.where(example_data == i)[0]])
    user_id = USERS[0] if i % 2 == 1 else USERS[1]
    embeds = dummy_embeds(img_ids, user_id, TRANSFORMATIONS[0])
    common_embeds = embeds[:NUMBER_OF_COMMON_EXAMPLES]
    other_embeds = embeds[NUMBER_OF_COMMON_EXAMPLES:]

    # add rows to the output dataframe
    for id, embed in zip(img_ids[:NUMBER_OF_COMMON_EXAMPLES], common_embeds):
        df_common_embeds.loc[len(df_common_embeds), :] = [id, embed, i]

    # add rows of not common images
    for id, embed in zip(img_ids[NUMBER_OF_COMMON_EXAMPLES:], other_embeds):
        df_other_embeds.loc[len(df_other_embeds), :] = [id, embed, i] 
    
    # sybil_attack_reset()


df_other_embeds.to_csv("other_embeddings.csv")
df_common_embeds.to_csv("common_embeddings.csv")

assert NUMBER_OF_COMMON_EXAMPLES + df_other_embeds.shape[1] == NUMBER_OF_EXAMPLES
