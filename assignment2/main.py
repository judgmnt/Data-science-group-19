import numpy as np
import pandas as pd
import time
import operator
from pprint import pprint

features = ["user_id","image_id","data_memorability","user_website","user_bio","user_followed_by","user_follows","user_posted_photos","face_gender","face_gender_confidence","face_age_range_high","face_age_range_low","face_sunglasses","face_beard","face_beard_confidence","face_mustache","face_mustache_confidence","face_smile","face_smile_confidence","eyeglasses","eyeglasses_confidence","face_emo","emo_confidence","comment_count","like_count","data_amz_label","data_amz_label_confidence"]
survey_df = pd.read_pickle(r'data/survey.pickle')
users = set(survey_df["insta_user_id"].apply(str))

# anp_df = pd.read_pickle(r'data/anp.pickle')

# survey_df = survey_df.astype({"insta_user_id": object})

# df = pd.merge(image_df, face_df, how='inner', on='image_id')
# df = pd.merge(df, metrics_df, how='inner', on='image_id')
# df = pd.merge(df, object_labels_df, how='inner', on='image_id')
# df = pd.merge(df, anp_df)

def get_user_image_data(return_user_id = True):
    image_df = pd.read_pickle(r'data/image_data.pickle')
    user_data = {}
    for index, row in image_df.iterrows():
        user_id = row["user_id"]
        if user_id not in users:
            continue

        if user_id not in user_data:
            user_data[user_id] = {}

        image_id = row["image_id"]
        if image_id not in user_data[user_id]:
            user_data[user_id][image_id] = np.zeros(4)

        user_data[user_id][image_id][0] += 1 if row["image_filter"] == "Normal" else 0

        user_data[user_id][image_id][1] = 1 if row["user_website"] else 0

        user_data[user_id][image_id][2] = row["user_followed_by"]

        user_data[user_id][image_id][3] = row["user_posted_photos"]

    columns = ["Filters", "Website", "Followers", "Follows"]
    return [user_data, columns]

def get_face_data():
    face_df = pd.read_pickle(r'data/face.pickle')

    emo_set = set(face_df["face_emo"])
    index = 0
    emo_list = {}

    for emo in emo_set:
        emo_list[emo] = index
        index += 1

    face_data = {}
    for index, row in face_df.iterrows():
        # if row["user_id"] not in users:
        #     continue

        image_id = row["image_id"]
        if image_id not in face_data:
            face_data[image_id] = np.zeros(15)

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
            face_data[image_id][8+emo_list[row["face_emo"]]] += 1

    # get average face age
    for id in face_data:
        face_data[id][2] = face_data[id][2]/(face_data[id][0]+face_data[id][1])

    # Create columns
    columns = ["Female", "Male", "Age", "Sunglasses", "Beard", "Mustache", "Smile", "Eyeglasses"]
    for emo in sorted(emo_list.items(), key=operator.itemgetter(1)):
        columns.append(emo[0])

    return [face_data, columns]

def get_metric_data():
    metrics_df = pd.read_pickle(r'data/image_metrics.pickle')
    metric_data = {}
    for index, row in metrics_df.iterrows():
        # if row["user_id"] not in users:
        #     continue

        image_id = row["image_id"]
        if image_id not in metric_data:
            metric_data[image_id] = np.zeros(2)
        
        metric_data[image_id][0] += row["like_count"]
        metric_data[image_id][1] += row["comment_count"]

    columns = ["Likes", "Comments"]
    return [metric_data, columns]

def get_object_count():
    object_labels_df = pd.read_pickle(r'data/object_labels.pickle')
    object_count = {}
    for index, row in object_labels_df.iterrows():
        # if row["user_id"] not in users:
        #     continue

        image_id = row["image_id"]
        if image_id not in object_count:
            object_count[image_id] = 0
        
        object_count[image_id] += 1
    columns = ["Objects"]
    return [object_count, columns]

# Easy function to filter DF based on columns
def make_xy(result_df, survey_df, col_subset = []):
    # col_subset is a list of columns to filter the DF by
    if len(col_subset) > 0:
        result_df = result_df[col_subset]

    x = []  #training data
    y = []  #label
    for index, row in result_df.iterrows():  #index is user_id
        # Y label
        label = survey_df[survey_df["insta_user_id"] == index]["PERMA"].values[0]
        if np.isnan(label): continue  #if the label doesn't exist (for the users with no perma score)
        y.append(label)

        # X label
        x.append(row)
    return (np.array(x), np.array(y))

def main():
    # Read data
    survey_df = pd.read_pickle(r'data/survey.pickle')
    survey_df = survey_df.astype({"insta_user_id": str})
    result_df = pd.read_pickle("result_df.pickle")

    # Transform to trainable data
    [x, y] = make_xy(result_df, survey_df)

    # Models
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    ### Train a regressor on all data ###
    
    # Random Forest ensemble
    regr = RandomForestRegressor(n_estimators=100)
    # SVM
    svr = SVR(gamma='scale', C=1.0, epsilon=0.2)

    # Cross validation
    k_fold = 10
    random_forest_total_score = []
    svm_total_score = []
    for i in range(k_fold):
        # Split into train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        
        # train models
        regr.fit(x_train, y_train)
        svr.fit(x_train, y_train)

        # Print R^2 scores of each model
        random_forest_score = regr.score(x_test, y_test)
        svm_result_score = svr.score(x_test, y_test)

        random_forest_total_score.append(random_forest_score)
        svm_total_score.append(svm_result_score)
        
        # Prints intermediate results
        print("K_fold ", i)
        # print("Tree: ", random_forest_score)
        # print("SVM: ", svm_result_score)

    # K fold cross validation results
    print("\nAverage Random Forest score: ", np.average(random_forest_total_score))
    print("Average SVM score: ", np.average(svm_total_score))
    

    ### Show Top features of decision tree
    print("\nFiltering based on top features ...")
    # Train decision tree on all data
    regr.fit(x, y)
    feature_list = []
    for index, col in enumerate(result_df.columns):
        feature_list.append([regr.feature_importances_[index], col])  
    
    n_features = 5
    top_n_features = np.array(sorted(feature_list))[:,1][-n_features:]
    bot_n_features = np.array(sorted(feature_list))[:,1][:n_features]
    print(top_n_features)
    print()

    # Make new dataframes based on top columns
    [x, y] = make_xy(result_df, survey_df, top_n_features)

    # Doing Cross validation again with filtered results
    k_fold = 10
    random_forest_total_score = []
    svm_total_score = []
    for i in range(k_fold):
        # Split into train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        
        # train models
        regr.fit(x_train, y_train)
        svr.fit(x_train, y_train)

        # Print R^2 scores of each model
        random_forest_score = regr.score(x_test, y_test)
        svm_result_score = svr.score(x_test, y_test)

        random_forest_total_score.append(random_forest_score)
        svm_total_score.append(svm_result_score)
        
        # Prints intermediate results
        print("K_fold ", i)
        # print("Tree: ", random_forest_score)
        # print("SVM: ", svm_result_score)

    # K fold cross validation results
    print("\nAverage Random Forest score: ", np.average(random_forest_total_score))
    print("Average SVM score: ", np.average(svm_total_score))



    ################## WARNING ####################
    ###### DONT LOOK AT THE CRAP BELOW HERE #######

    #################### CREATE FINAL DF
    # [total_data, total_cols] = get_user_image_data()
    # df = df.fillna(0)

    # index = 0
    # result = {}
    # for user in total_data:
    #     print("User: ", user)
    #     total_rows = []
    #     for image_id in total_data[user]:
    #         row = list(df[df.index == image_id].values.flatten())
    #         if len(row) < 1: continue
    #         for element in total_data[user][image_id]:
    #             row.append(element)
    #         total_rows.append(row)
    #     if not row: continue
    #     result[user] = np.average(total_rows, axis=0)
        
    #     print(index)
    #     index += 1

    # columns = list(df.columns)
    # for col in total_cols:
    #     columns.append(col)
    # result_df = pd.DataFrame.from_dict(result, orient='index')
    # result_df.columns = columns
    # result_df.to_pickle("result_df.pickle")
    # print(result_df[:10])

    ################ CREATE DICTS

    # [metric_data, metric_cols] = get_metric_data()
    # df1 = pd.DataFrame.from_dict(metric_data, orient='index')
    # df1.columns = metric_cols
    # print("1", len(df1))
    
    # [face_data, face_cols] = get_face_data()
    # df2 = pd.DataFrame.from_dict(face_data, orient='index')
    # df2.columns = face_cols
    # print("2", len(df2))

    # [object_data, object_cols] = get_object_count()
    # df3 = pd.DataFrame.from_dict(object_data, orient='index')
    # df3.columns = object_cols
    # print("3", len(df3))

    # df = df1.merge(df2, how='outer', left_index=True, right_index=True)
    # df = df.merge(df3, how='outer', left_index=True, right_index=True)
    # print(df.shape)
    # print(len(df))
    # print(df.columns)

    # df.to_pickle("image_df.pickle")


if __name__ == '__main__':
    main()

# Feature list
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


