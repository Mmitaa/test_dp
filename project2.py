
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import log_softmax, nll_loss
#from google.colab import drive
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

"""## Загрузка ответов ученика и эталонных ответов
---
"""

# Монтирование Google Drive
#drive.mount('/content/drive')

path = "D:/Загрузки/pd/pd/"
modelpath = "D:/Загрузки/pd/pd/"

##path = "/content/drive/My Drive/Colab Notebooks/NNet&App/2_3 NLP/data/"
#modelpath = "/content/drive/My Drive/Colab Notebooks/NNet&App/2_3 NLP/models/"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Загрузка вопросов
def get_data_from_csv_file(path_to_file):
    return pd.read_csv(path_to_file, index_col=0, sep=';', header=None, encoding='utf-8')

def get_dictionary_with_text(path_to_file):
    data = get_data_from_csv_file(path_to_file)
    data.columns = ['text']
    data_dictionary = {id: data.loc[id, 'text'] for id in data.index}
    return data_dictionary

def get_dictionary_with_id(path_to_file):
    data = get_data_from_csv_file(path_to_file)
    data.columns = ['answer_id']
    data_dictionary = {id: [int(key) for key in data.loc[id, 'answer_id'].split(',')] for id in data.index}
    print(data_dictionary)
    return data_dictionary

def numpy_array_to_dict(numpy_array):
    dictionary = {}
    rows, cols = numpy_array.shape
    for i in range(rows):
        for j in range(cols):
            key = f'row_{i}_col_{j}'
            value = numpy_array[i, j]
            dictionary[key] = value
    return dictionary

def get_dictionary_with_id__(path_to_file):
    data = get_data_from_csv_file(path_to_file)
    data.columns = ['answer_id']
    data_dictionary = {id: [int(data.loc[id, 'answer_id'])] for id in data.index}
    print(data_dictionary)
    return data_dictionary

def get_dictionary_with_id__(path_to_file):
    data = get_data_from_csv_file(path_to_file)
    data.columns = ['answer_id']
    data_dictionary = {id: list(map(int, data.loc[id, 'answer_id'].split(','))) for id in data.index}
    print(data_dictionary)
    return data_dictionary

def get_index_dictionary(data):
    result = {index: item for index, item in enumerate(data)}
    return result

questions_dict = get_dictionary_with_text(path + "questions.csv")
answers_dict = get_dictionary_with_text(path + "answers.csv")

half_relations_dict = get_dictionary_with_id(path + "half_relations.csv")
neg_relations_dict = get_dictionary_with_id(path + "neg_relations.csv")

ref_relations_dict = get_dictionary_with_id__(path + "ref_relations.csv")
right_relations_dict = get_dictionary_with_id(path + "right_relations.csv")

print(len(questions_dict), len(answers_dict))  # Выводим длину словарей вопросов и ответов

ref_relations = pd.read_csv(path + "ref_relations.csv", sep=';', header=None)
right_relations = pd.read_csv(path + "right_relations.csv", sep=';', header=None)
half_relations = pd.read_csv(path + "half_relations.csv", sep=';', header=None)
neg_relations = pd.read_csv(path + "neg_relations.csv", sep=';', header=None)
ref_relations.tail(3)

count = 0
for id, question in questions_dict.items():
    print(f"id={id}: {question}")
    count += 1
    if count > 3:
        break

count = 0
for id, answer in answers_dict.items():
    print(f"id={id}: {answer}")
    count += 1
    if count > 3:
        break


"""## Формирование эмбеддингов эталонных ответов и ответов ученика
---
"""

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_bert_cls(text, model, tokenizer):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings[0].cpu().numpy()

label_negative = 0
# Загрузка модели LaBSE
tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

answers_vector = {id: embed_bert_cls(answer, model, tokenizer) for id, answer in answers_dict.items()}

count = 0
for id, ans in answers_vector.items():
    print(f"id={id}: {ans[:5]}...")
    count += 1
    if count > 5:
        break

"""## Формирование выборок для обучения

---


---
"""

# Создание обучающей выборки
X = []
y = []

label_positive = 1  # Правильный ответ
label_negative = 0  # Неправильный ответ

# Добавление пары вида эталонный ответ - правильный ответ
for question_id, right_answers_ids in right_relations_dict.items():
    ref_answer_ids = ref_relations_dict[question_id]
    ref_answers = [answers_vector[ref_id] for ref_id in ref_answer_ids]
    for ref_answer in ref_answers:
        for right_answer_id in right_answers_ids:
            right_answer = answers_vector[right_answer_id]
            similarity = cosine_similarity([right_answer], [ref_answer])[0][0]
            X.append(np.concatenate((right_answer, ref_answer, [similarity])))
            y.append(label_positive)


# Добавляем пары вида эталонный ответ - неполный ответ
for question_id, half_answers_ids in half_relations_dict.items():
    ref_answer_ids = ref_relations_dict[question_id]
    ref_answers = [answers_vector[ref_id] for ref_id in ref_answer_ids]
    for ref_answer in ref_answers:
        for half_answer_id in half_answers_ids:
            half_answer = answers_vector[half_answer_id]
            similarity = cosine_similarity([half_answer], [ref_answer])[0][0]
            X.append(np.concatenate((half_answer, ref_answer, [similarity])))
            y.append(label_negative)

# Добавляем пары вида эталонный ответ - неправильный ответ
for question_id, neg_answers_ids in neg_relations_dict.items():
    ref_answer_ids = ref_relations_dict[question_id]
    ref_answers = [answers_vector[ref_id] for ref_id in ref_answer_ids]
    for ref_answer in ref_answers:
        for neg_answer_id in neg_answers_ids:
            neg_answer = answers_vector[neg_answer_id]
            similarity = cosine_similarity([neg_answer], [ref_answer])[0][0]
            X.append(np.concatenate((neg_answer, ref_answer, [similarity])))
            y.append(label_negative)


X = np.array(X)
y = np.array(y)

X.shape, y.shape

X[0].shape, X[0],  y[0]

X[-1].shape, X[-1], y[-1]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) #, random_state=RANDOM_SEED)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#criterion = nn.CrossEntropyLoss()
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Создание класса
class ImprovedNN(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128, 32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(32, 1)
        # self.relu4 = nn.ReLU()
        # self.fc5 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        # x = self.relu4(x)
        # x = self.fc5(x)
        x = self.sigmoid(x)
        return x

# Инициализация нейросети, функции потерь и оптимизатора
input_size = X_train.shape[1]
model = ImprovedNN(input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model

# Обучение нейросети
num_epochs = 100
for epoch in range(num_epochs):
    inputs = Variable(torch.from_numpy(X_train)).float()
    labels = Variable(torch.from_numpy(y_train)).float()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels.view(-1, 1))
    loss.backward()
    optimizer.step()
    #scheduler.step()

    predicted_labels = (outputs >= 0.5).float()
    accuracy = (predicted_labels == labels.view(-1, 1)).float().mean().item()

    if epoch % 10 == 0:
         print(f'Epoch {epoch}, Loss: {loss.item()} , Accuracy: {accuracy * 100}%')

# Оценка точности на тестовой выборке
model.eval()
with torch.no_grad():
    test_inputs = Variable(torch.from_numpy(X_test)).float()
    test_labels = Variable(torch.from_numpy(y_test)).float()
    test_outputs = model(test_inputs)
    predicted_labels = (test_outputs >= 0.5).float()
    accuracy1 = (predicted_labels == test_labels.view(-1, 1)).float().mean().item()

print(f'Accuracy: {accuracy1 * 100}%')

import random
rand = random.randint(1,2)
mmodel = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
rand_question_id = random.choice(list(questions_dict.keys()))
print(rand_question_id, questions_dict[rand_question_id])
answer = input("Ваш ответ: ")
new_answer = embed_bert_cls(answer, mmodel, tokenizer)
ref_relation_ids = ref_relations_dict[rand_question_id]
right_relation_ids = right_relations_dict[rand_question_id]
ref_relation = answers_vector[ref_relation_ids[0]]
similarity = cosine_similarity([new_answer], [ref_relation])[0][0]
new_input = np.concatenate((new_answer, ref_relation, [similarity]))
model.eval()
with torch.no_grad():
    test_input = Variable(torch.from_numpy(np.asarray(new_input))).float()
    test_labels = Variable(torch.from_numpy(ref_relation)).float()
    test_output = model(test_input)
    print(test_output)
    print("Эталонный ответ:", answers_dict[ref_relation_ids[0]])
    print("Правильный ответ: ", answers_dict[right_relation_ids[0]])


