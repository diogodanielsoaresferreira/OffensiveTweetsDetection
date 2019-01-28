import pandas
from sklearn.model_selection import train_test_split


input_file = "Dataset for Detection of Cyber-Trolls.json"
train_output_file = "train_data.json"
test_output_file = "test_data.json"
test_percentage = 0.2

# Load json data as records and remove columns where all values are NaN
data = pandas.read_json(input_file, lines=True, orient="records").dropna(axis=1, how='all')

# Change annotation column to contain label
data = data.join(data.pop('annotation').apply(lambda x: x['label'][0]))
data = data.rename(columns={data.columns[1]: "label"})

X_train, X_test, y_train, y_test = train_test_split(data["content"], data["label"], test_size=test_percentage, shuffle=True)

train_data = pandas.DataFrame({"content": X_train, "label": y_train})
test_data = pandas.DataFrame({"content": X_test, "label": y_test})
train_data.to_json(train_output_file, lines=True, orient="records")
test_data.to_json(test_output_file, lines=True, orient="records")
