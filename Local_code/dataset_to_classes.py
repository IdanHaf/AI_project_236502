import pandas as pd


def is_unitable(mean_coords, mean_coords2, lat_dist):
    lat, lng = mean_coords[0], mean_coords[1]
    lat2, lng2 = mean_coords2[0], mean_coords2[1]

    lat_distance = abs(lat - lat2)
    lng_distance = abs(lng - lng2)

    return lat_distance <= lat_dist and lng_distance <= lat_dist*2


# Load CSV data
df = pd.read_csv('coords.csv', header=None)
classes_sets = [([i], df.iloc[i].tolist()) for i in range(len(df))]

visited = []

dist = [3, 5, 6]

for lat_dist in dist:
    for index in range(len(df)):
        if index in visited:
            continue

        if index % 10 == 0:
            print(index)

        cls_set, mean_coords = classes_sets[index]

        for j in range(index + 1, len(df)):
            if j in visited:
                continue

            cls_set2, mean_coords2 = classes_sets[j]

            if is_unitable(mean_coords, mean_coords2, lat_dist):
                mean_coords[0] = (mean_coords[0] * len(cls_set) + mean_coords2[0] * len(cls_set2)) / (
                        len(cls_set) + len(cls_set2))
                mean_coords[1] = (mean_coords[1] * len(cls_set) + mean_coords2[1] * len(cls_set2)) / (
                        len(cls_set) + len(cls_set2))

                cls_set += cls_set2
                visited.append(j)


print("finished")

model_classes = []
mean_model_class_coords = []
samples_in_group = 0

for index, (cls_set, mean_coords) in enumerate(classes_sets):
    if index in visited:
        continue

    print(index)
    print(len(cls_set))
    samples_in_group += len(cls_set)
    model_classes.append(cls_set)
    mean_model_class_coords.append(mean_coords)
    print("mean_model_class_coords:")
    print(mean_model_class_coords)

print(len(visited))
print(len(set(visited)))
print("samples_in_group: ", samples_in_group)

# Save in csv different classes
result_df = pd.DataFrame(0, index=df.index, columns=[0])
result_df_mean_coords = pd.DataFrame(mean_model_class_coords)

class_num = 0
for mdl_cls in model_classes:
    result_df.loc[mdl_cls, 0] = class_num
    class_num += 1

print("save as csv")
result_df.to_csv('result.csv', header=False, index=False)
result_df_mean_coords.to_csv('result_mean_coords.csv', header=False, index=False)
print("file saved")
