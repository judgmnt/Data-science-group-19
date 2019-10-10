import numpy as np
import pandas as pd
import time
from pprint import pprint

features = ["user_id","image_id","data_memorability","user_website","user_bio","user_followed_by","user_follows","user_posted_photos","face_gender","face_gender_confidence","face_age_range_high","face_age_range_low","face_sunglasses","face_beard","face_beard_confidence","face_mustache","face_mustache_confidence","face_smile","face_smile_confidence","eyeglasses","eyeglasses_confidence","face_emo","emo_confidence","comment_count","like_count","data_amz_label","data_amz_label_confidence"]
# anp_df = pd.read_pickle(r'data/anp.pickle')
face_df = pd.read_pickle(r'data/face.pickle')
image_df = pd.read_pickle(r'data/image_data.pickle')
metrics_df = pd.read_pickle(r'data/image_metrics.pickle')
object_labels_df = pd.read_pickle(r'data/object_labels.pickle')
# survey_df = pd.read_pickle(r'data/survey.pickle')

# survey_df = survey_df.astype({"insta_user_id": object})

# df = pd.merge(image_df, face_df, how='inner', on='image_id')
# df = pd.merge(df, metrics_df, how='inner', on='image_id')
# df = pd.merge(df, object_labels_df, how='inner', on='image_id')
# df = pd.merge(df, anp_df)

users = set(image_df["user_id"])
data = {}
for id in users:
    if id not in data:
        data[id] = np.zeros(5)

emo_set = set(face_df["face_emo"])
index = 0
emo_list = {}
for emo in emo_set:
    emo_list[emo] = index
    index += 1

face_data = {}
for index, row in face_df.iterrows():

    image_id = row["image_id"]
    if image_id not in face_data:
        face_data[image_id] = np.zeros(16)

    # 1-2 female male
    if row["face_gender"] == "Female":
        face_data[image_id][0] += 1
    else:
        face_data[image_id][1] += 1

    # 3 - face age
    face_data[image_id][2] += (row["face_age_range_low"]+row["face_age_range_high"])/2

    # 4 - sunglasses
    face_data[image_id][3] += 1 if row["face_sunglasses"] else 0

    # 5 - beard
    face_data[image_id][4] += 1 if row["face_beard"] and row["face_beard_confidence"] > 90. else 0

    # 6 - mustache
    face_data[image_id][5] += 1 if row["face_mustache"] and row["face_mustache_confidence"] > 90. else 0

    # 7 - smile
    face_data[image_id][6] += 1 if row["face_smile"] and row["face_smile_confidence"] > 90. else 0

    # 8 - eyeglasses
    face_data[image_id][7] += 1 if row["eyeglasses"] and row["eyeglasses_confidence"] > 90. else 0

    # 9 - 15
    if row["emo_confidence"] > 60.:
        face_data[image_id][emo_list[row["face_emo"]]] += 1
    break

for id in face_data:
    face_data[id][2] = face_data[id][2]/(face_data[id][0]+face_data[id][1])

metric_data = {}
for index, row in metrics_df.iterrows():
    image_id = row["image_id"]
    if image_id not in metric_data:
        metric_data[image_id] = np.zeros(2)
    
    metric_data[image_id][0] += row["like_count"]
    metric_data[image_id][1] += row["comment_count"]

object_count = {}
for index, row in object_labels_df.iterrows():
    image_id = row["image_id"]
    if image_id not in object_count:
        object_count[image_id] = 0
    
    object_count[image_id] += 1

# # anp_df 325941 ['image_id', 'anp_label', 'anp_sentiment', 'emotion_score', 'emotion_label']
# print("\nanp_df", len(anp_df), list(anp_df.columns))

# # face_df 86877 ['image_id', 'face_id', 'face_gender', 'face_gender_confidence', 'face_age_range_high', 'face_age_range_low', 'face_sunglasses', 'face_beard', 'face_beard_confidence', 'face_mustache', 'face_mustache_confidence', 'face_smile', 'face_smile_confidence', 'eyeglasses', 'eyeglasses_confidence', 'face_emo', 'emo_confidence']
# print("\nface_df", len(face_df), list(face_df.columns))

# # image_df 41206 ['image_id', 'image_link', 'image_url', 'image_height', 'image_width', 'image_filter', 'image_posted_time_unix', 'image_posted_time', 'data_memorability', 'user_id', 'user_full_name', 'user_name', 'user_website', 'user_profile_pic', 'user_bio', 'user_followed_by', 'user_follows', 'user_posted_photos']
# print("\nimage_df", len(image_df), list(image_df.columns))

# # metrics_df 44218 ['image_id', 'comment_count', 'comment_count_time_created', 'like_count', 'like_count_time_created']
# print("\nmetrics_df", len(metrics_df), list(metrics_df.columns))

# # object_labels_df 172613 ['image_id', 'data_amz_label', 'data_amz_label_confidence']
# print("\nobject_labels_df", len(object_labels_df), list(object_labels_df.columns))

# # survey_df 161 ['index', 'id', 'gender', 'born', 'education', 'employed', 'income', 'A_2', 'N_1', 'P_1', 'E_1', 'A_1', 'H_1', 'M_1', 'R_1', 'M_2', 'E_2', 'LON', 'H_2', 'P_2', 'N_2', 'A_3', 'N_3', 'E_3', 'H_3', 'R_2', 'M_3', 'R_3', 'P_3', 'HAP', 'participate', 'insta_user_id', 'completed', 'start_q', 'end_q', 'network_id', 'P', 'E', 'R', 'M', 'A', 'PERMA', 'N_EMO', 'P_EMO', 'imagecount', 'private_account']
# print("\nsurvey_df", len(survey_df), list(survey_df.columns))


