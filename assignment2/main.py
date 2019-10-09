import pandas as pd
from pprint import pprint

features = ["image_filter","data_memorability","user_website","user_bio","user_followed_by","user_follows","user_posted_photos","anp_label","anp_sentiment","emotion_score","emotion_label","data_amz_label","data_amz_label_confidence","face_gender","face_gender_confidence","face_age_range_high","face_age_range_low","face_sunglasses","face_beard","face_beard_confidence","face_mustache","face_mustache_confidence","face_smile","face_smile_confidence","eyeglasses","eyeglasses_confidence","face_emo","emo_confidence"]
anp_df = pd.read_pickle(r'data/anp.pickle')
face_df = pd.read_pickle(r'data/face.pickle')
image_df = pd.read_pickle(r'data/image_data.pickle')
metrics_df = pd.read_pickle(r'data/image_metrics.pickle')
object_labels_df = pd.read_pickle(r'data/object_labels.pickle')
survey_df = pd.read_pickle(r'data/survey.pickle')

# image_anp_frame = pd.merge(image_df, anp_df, how='inner', on='image_id')
# im_anp_obj_frame = pd.merge(image_anp_frame, object_labels_df, how='inner', on='image_id')
# im_anp_obj_face_frame = pd.merge(im_anp_obj_frame, face_df, how='inner', on='image_id')
# im_anp_obj_face_frame = pd.merge(im_anp_obj_frame, face_df, how='inner', on='image_id')

print("\nanp_df", len(anp_df), list(anp_df.columns))
print("\nface_df", len(face_df), list(face_df.columns))
print("\nimage_df", len(image_df), list(image_df.columns))
print("\nmetrics_df", len(metrics_df), list(metrics_df.columns))
print("\nobject_labels_df", len(object_labels_df), list(object_labels_df.columns))
print("\nsurvey_df", len(survey_df), list(survey_df.columns))
