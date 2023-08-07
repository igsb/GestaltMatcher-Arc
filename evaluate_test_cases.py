import os
import json
import numpy as np
import pandas as pd

results_dir = 'test_results'
image_results_dict = {}

label_list = []
dist_list = []

# parse syndrome
synd_df = pd.read_csv('data/GestaltMatcherDB/v1.0.3/gmdb_metadata/gmdb_syndromes_v1.0.3.tsv', sep='\t')
synd_dict = {row['syndrome_id']: row['syndrome_name'] for _, row in synd_df.iterrows()}

image_df = pd.read_csv('data/GestaltMatcherDB/v1.0.3/gmdb_metadata/gmdb_test_images_v1.0.3.csv', sep=',')
image_to_synd_dict = {row['image_id']: synd_dict[row['label']] for _, row in image_df.iterrows()}

for i in os.listdir(results_dir):
    result_file = os.path.join(results_dir, i)
    f = open(result_file)
    data = json.load(f)
    case_id = int(i.split('.')[0])
    target_synd = image_to_synd_dict[case_id]
    # list
    count = 1
    for j in data['suggested_syndromes_list']:
        if j['syndrome_name'] == target_synd:
            image_results_dict[i.split('.')[0]] = count
            dist_list.append(j['distance'])
        count += 1

    f.close()

ranks = np.array([i for _, i in image_results_dict.items()])
top_ranks_list = [sum(ranks < i)/len(ranks)*100 for i in range(2, 52)]
for i in [0, 4, 9, 29]:
    print("Top {}: {}%".format(i+1, top_ranks_list[i]))

#dist_list = np.array(dist_list)
#print(sum(dist_list<0.4))
