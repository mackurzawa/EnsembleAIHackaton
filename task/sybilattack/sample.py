import numpy as np
import pandas as pd
import time
import torch
from sybil_requests import sybil, sybil_reset
from taskdataset import TaskDataset

NUMBER_OF_EXAMPLES = 20000
NUMBER_OF_COMMON_EXAMPLES = 615
NUMBER_OF_CLUSTERS = 14

USERS = ["home", "defense"]
TRANSFORMATIONS = ["binary", "affine"]


df_common_embeds = pd.DataFrame(columns=["img_id", "embedding", "transformation"])
df_other_embeds = pd.DataFrame(columns=["img_id", "embedding", "transformation"])


dataset = torch.load("task/sybilattack/data/SybilAttack.pt")
common_img_ids = [d[0] for d in zip(*dataset[:NUMBER_OF_COMMON_EXAMPLES])]

example_data = np.zeros(NUMBER_OF_EXAMPLES, dtype=int)

other_imgs_ids = np.array([d[0] for d in zip(*dataset[NUMBER_OF_COMMON_EXAMPLES:])])
for i in range(NUMBER_OF_COMMON_EXAMPLES, NUMBER_OF_EXAMPLES):
    example_data[i] = i%14 + 1


sybil_reset(TRANSFORMATIONS[1], USERS[0])
sybil_reset(TRANSFORMATIONS[1], USERS[1])
for i in range(1, NUMBER_OF_CLUSTERS+1):
    img_ids = np.concatenate([common_img_ids, other_imgs_ids[example_data[NUMBER_OF_COMMON_EXAMPLES:] == i]])
    user_id = USERS[0] if i % 2 == 1 else USERS[1]
    
    while True:
        try:
            embeds = sybil(img_ids, user_id, TRANSFORMATIONS[1])
            break
        except Exception as e:
            time.sleep(0.5)

    common_embeds = embeds[:NUMBER_OF_COMMON_EXAMPLES]
    other_embeds = embeds[NUMBER_OF_COMMON_EXAMPLES:]

    # add rows to the output dataframe
    for id, embed in zip(img_ids[:NUMBER_OF_COMMON_EXAMPLES], common_embeds):
        df_common_embeds.loc[len(df_common_embeds), :] = [id, embed, i]


    # add rows of not common images
    for id, embed in zip(img_ids[NUMBER_OF_COMMON_EXAMPLES:], other_embeds):
        df_other_embeds.loc[len(df_other_embeds), :] = [id, embed, i] 
    
    sybil_reset(TRANSFORMATIONS[1], user_id)


df_other_embeds.to_csv(f"other_embeddings_{TRANSFORMATIONS[1]}.csv", index=False)
df_common_embeds.to_csv(f"common_embeddings_{TRANSFORMATIONS[1]}.csv", index=False)
