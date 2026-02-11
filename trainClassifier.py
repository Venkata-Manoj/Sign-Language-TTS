import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

MIN_SAMPLES = 30

label_counts = Counter(labels)

valid_indices = [
    i for i, label in enumerate(labels)
    if label_counts[label] >= MIN_SAMPLES
]

data = np.array(data)[valid_indices]
labels = np.array(labels)[valid_indices]

print("Filtered class counts:", Counter(labels))

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model_isl.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
