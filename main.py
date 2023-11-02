import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
data_file = 'datasets_420115_802391_bmi_data.csv'
# Load data and analyse (Tải dữ liệu và phân tích)
df = pd.read_csv(data_file)
df.columns = ['Sex', 'Age', 'Height', 'Weight', "BMI"]

# 3 Drop missing rows (bỏ các hàng bị thiếu)
df = df.dropna()
df.head(5)  # 5 column
print(df)  # in ra bang datatset

# 4 Get an overview of data (Nhận tổng quan về dữ liệu)
df.describe()

# 5
colors = [(1 - (BMI - 13) / 14, 0, 0) for BMI in df.BMI.values]
fig, ax = plt.subplots()
ax.scatter(df['Weight'].values, df['Height'].values, c=colors, picker=True)
ax.set_xlabel('Weight')
ax.set_ylabel('Height')
_ = ax.set_title('BMI distribution')

# Huấn luyện mô hình tuyến tính để dự đoán chỉ số BMI dựa trên độ tuổi, chiều cao và cân nặng\
# BMI = W1 * height + W2 * weight + w3 * age +bias
# Chia thành train and validation
train_pct = 0.8
train_index = int(len(df) * train_pct)

train_data = df.iloc[:train_index].copy()
validation_data = df.iloc[train_index:].copy()
print(f'train = {len(train_data)},\nvalidation = {len(validation_data)}')


# Khởi tạo ngẫu nhiên các trọng số
def reset():
    global w1, w2, w3, bias
    w1 = np.random.randn()
    w2 = np.random.randn()
    w3 = np.random.randn()
    bias = np.random.randn()

reset()

print_weight = lambda: print('w1 = {},\nw2 = {},\nw3 = {},\nbias = {}'.format(w1, w2, w3, bias))
print_weight()

# Chuẩn hóa các tính năng
# Sửa đổi các tính năng đầu vào và BMI đầu ra để có giá trị trung bình=0 độ lệch chuẩn = 1
# 8
def normalize(df, means, stds):
    #print(means)
    df['Weight'] = (df['Weight'] - means.Weight) / stds.Weight
    df['Height'] = (df['Height'] - means.Height) / stds.Height
    df['Age'] = (df['Age'] - means.Age) / stds.Age
    if 'BMI' in df.columns:
        df['BMI'] = (df['BMI'] - means.BMI) / stds.BMI
    df.head()
    return df


def de_normalize(df, means, stds):
    # print(means)
    df = df.copy()
    df['Weight'] = df['Weight'] * stds.Weight + means.Weight
    df['Height'] = df['Height'] * stds.Height + means.Height
    df['Age'] = df['Age'] * stds.Age + means.Age
    if 'BMI' in df.columns:
        df['BMI'] = df['BMI'] * stds.BMI + means.BMI
    if 'predictionBMI' in df.columns:
        df['predictionBMI'] = df['predictionBMI'] * stds.BMI + means.BMI

    return df

# 9
means = train_data.iloc[:, 1:].mean()
stds = train_data.iloc[:, 1:].std()
normalize(train_data, means, stds)
print('Normalized train data')
train_data.head()

# 10
normalize(validation_data, means, stds)
print('Normalized test data')
validation_data.head()


# Dự đoán BMI bằng hàm tuyến tính
# 11
def predict_BMI(df):
    pred = w1 * df['Height'] + w2 * df['Weight'] + w3 * df['Age'] + bias
    df['predictionBMI'] = pred
    return df

print('Random weights predictions')
preddf = predict_BMI(train_data)
preddf.head()

#12 Loss function
def calculate_loss(df):
    return np.square(df['predictionBMI'] - df['BMI'])

preddf = predict_BMI(train_data)
print('loss = ', calculate_loss(preddf).mean())

#13
def calculate_gradients(df):
    diff = df['predictionBMI'] - df['BMI']
    dw1 = 2 * diff *df['Height']
    dw2 = 2 * diff *df['Weight']
    dw3 = 2 * diff *df['Age']
    dbias = 2* diff
    dw1,dw2,dw3 , dbias  =  dw1.values.mean(),dw2.values.mean(),dw3.values.mean(),dbias.values.mean()
    #print(dw1,dw2,dw3 , dbias)
    return dw1,dw2,dw3 , dbias

#14
def train(learning_rate = 0.01):
    global w1, w2, w3, bias, preddf
    dw1,dw2,dw3 , dbias = calculate_gradients(preddf)
    w1 = w1 - dw1*learning_rate
    w2 = w2 - dw2 * learning_rate
    w3 = w3 - dw3 * learning_rate
    bias = bias - dbias.mean() * learning_rate
    #print(w1, w2, w3, bias)
    preddf = predict_BMI(train_data)
    return calculate_loss(preddf).mean()

#15
print('\nPrediction on validation set before training')  #Dự đoán về bộ xác thực trước khi đào tạo
de_normalize(predict_BMI(validation_data),means,stds).head(10)

#Train
#16
import time
import math
from tqdm import  tqdm

learning_rate = 0.01

for i in tqdm(range(300)):
    loss = train(learning_rate)
    time.sleep(0.01)
    if i%20 ==0:
        print(f'epoch : {i} : loss = {loss}')

#17
print('after training')
print_weight()

#18
print('\nPrediction on validation set after training')
de_normalize(predict_BMI(validation_data),means,stds).head(10)

#19
def predictBMI_real(data):
    df = pd.DataFrame(data)
    normalize(df,means, stds)
    return de_normalize(predict_BMI(df),means, stds)

#Sử dụng mô hình tuyến tính tính chỉ số BMI của tôi
#20
new_data = [{'name' :'Krishan', 'Age': 30, 'Height': 68, 'Weight': 157.63}]
newBMI = predictBMI_real(new_data)
print(newBMI)
#print(predictBMI_real(df))

plt.show()
