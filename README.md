# Decision Support System: Model-based Collaborative Filtering (User Clustering + Slope One)

## Design of the Recommendation System
![image](Design%20of%20the%20Recommendation%20System.png)

## Implementation of Slope One Algorithm
SLOPE ONE: Method used for Collaborative Filtering that is easy to implement efficiently and with a precision similar to algorithms with greater complexity, difficult to implement and expensive to train.

- Input: A DATAFRAME with 3 columns (USERID, ITEMID and RATING)
- Process:
  - The input is transformed into an array of ITEMS and USERS with the RATINGS as values.
  - The items are iterated sequentially and nested to find the average difference between each ITEM; For this, users who have rated both ITEMS are searched for.
  - A symmetric matrix of average differences (DEVIATIONS) is generated, where the lower side of the diagonal represents the negative average difference due to the change in the order of the ITEMS compared.
- Output: Two DATAFRAMES.
  - The first is an array of ITEMS (columns) and USERS (rows) with the RATINGS as values.
  - The second is an array of ITEMS (columns and rows) with the DEVIATIONS as values.
 
```
def one_slope(dataframe):
    rating_matrix = dataframe.pivot_table(index='userId', columns='itemId', values='rating')
    def get_dev_val(dataframe, i, j):
        dataframe = dataframe[
            (dataframe[i].notnull())
            & (dataframe[j].notnull())
        ].copy()
        users = len(dataframe)
        if users > 0: dev_val = (rating_matrix[i]-rating_matrix[j]).sum() / users
        else: dev_val = 0
        return dev_val

    deviations = {}
    for i in rating_matrix.columns:
        for j in rating_matrix.columns:
            if i == j:
                continue
            deviation = get_dev_val(rating_matrix, i, j)
            deviations[i, j] = deviation
            deviations[j, i] = (-1) * deviation

    desviation_matrix = get_matrix(deviations)
    return rating_matrix, desviation_matrix
```
 
## Application of Collaborative Filtering
When the groups of users are identified according to the characterization of their variables, each of the groups is iterated to filter the users of that group and obtain their rating and deviation matrices; This process is performed for each pool. During the process, the results obtained are stored and the data structure used for this activity is a dictionary.

```
matrix_per_cluster = {}
for cluster in df_user_clustered['cluster'].unique():
    matrix = {}
    users_clustered = df_user_clustered[df_user_clustered['cluster'] == cluster]['userId'].tolist()
    df_ratings_clustered = df_ratings_all[df_ratings_all['userId'].isin(users_clustered)]
    matrix['rating_matrix'], matrix['desviation_matrix'] = one_slope(df_ratings_clustered)
    matrix_per_cluster[cluster] = matrix
```

## Predict Movie Ratings
ITEMS (movies) will be recommended to 100 users. In this process, each one of the users is iterated:
- For each user, the ONE SLOPE algorithm is used to calculate the possible RATINGS that the evaluated user will assign to ITEMS that he has not rated (it is assumed that by not rating it, he has not seen it).
- For this calculation, the DEVIATIONS matrix and the already qualified elements are used. For each element evaluated, a value is assigned based on those already qualified with the corresponding deviation. These values are then averaged to assign a single possible rating to the movie.
- Movies that the user has already rated are not recommended. The rating assigned to these elements is NULL.
- The result is an array of CALCULATED RATINGS.

```
user_to_predict = df_ratings_known['userId'].unique().tolist()
predictions = {}
for u in user_to_predict:
    user_cluster = df_user_clustered[df_user_clustered['userId'] == u]['cluster'].values[0]
    user_matrix = matrix_per_cluster[user_cluster]
    user_rating_matrix = user_matrix['rating_matrix']
    user_desviation_matrix = user_matrix['desviation_matrix']
    user_row = user_rating_matrix.loc[u, :]
    user_ranked_items = user_row[user_row.notnull()].index
    for i in user_desviation_matrix.columns:
        if i not in user_ranked_items:
            predictions[u, i] = (np.sum(user_rating_matrix.loc[u, user_ranked_items] + user_desviation_matrix.loc[i, user_ranked_items])) / len(user_ranked_items)
predictions_matrix = get_matrix(predictions)
```

## Movie Recommendation
The best predictions are chosen based on the predicted RATING of the ITEMS (movies) without qualifying for the 100 users; for this:
- The users of the predicted RATINGS array are iterated.
- ITEMS (movies) are ordered from highest to lowest.
- The 10 films with the highest score are chosen.
- They are stored in a dictionary, where each key is a user and each value is a list of 10 tuples (ITEM, RATING) with the highest rating score.

```
best_predictions = {}
for idx, row in predictions_matrix.iterrows():
    top_10 = row.sort_values(ascending=False).head(10)
    best_predictions[idx] = list(zip(top_10.index, top_10.values))
```

## Evaluation with Qualitative Metrics
- DCG: The graded relevance value decreases logarithmically proportional to the position of the result.
```
def discounted_cumulative_gain(recommended_list, ground_truth_list):
    dcg_value = 0
    asserts = recommended_list[recommended_list['itemId'].isin(ground_truth_list['itemId'])]['itemId'].tolist()
    for assert_item in asserts:
        gt_item_position = ground_truth_list[ground_truth_list['itemId'] == assert_item].index[0]
        r_item_position = recommended_list[recommended_list['itemId'] == assert_item].index[0]
        gt_item_rating = ground_truth_list.loc[gt_item_position, 'rating']
        dcg_value += gt_item_rating / np.log2(r_item_position + 2) #se agrega un uno adicional porque empieza en 0
    return dcg_value
```

- IDCG: The graded relevance value decreases logarithmically proportional to the relative position of the user-assigned value.
```
def ideal_discounted_cumulative_gain(ground_truth_list):
    idcg_value = 0
    for item in ground_truth_list.index:
        idcg_value += ground_truth_list.loc[item, 'rating'] / np.log2(item + 2) #se agrega un uno adicional porque empieza en 0
    return idcg_value
```
    
- NDCG: Normalization of the metric based on the division of the DCG and the IDCG.
```
normalize_discounted_cumulative_gain = discounted_cumulative_gain / ideal_discounted_cumulative_gain
```

