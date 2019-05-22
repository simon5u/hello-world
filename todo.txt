#### numpy

import numpy as np
image = np.zeros((256, 256, 3), dtype=np.float32)

#### Pandas

import pandas as pd
from pandas import Series, DataFrame


### csv读取
dtype = {
    'id': np.int8,
    'qid1': np.int8,
    'qid2': np.int8,
    'question1': np.str,
    'question2': np.str,
    'is_duplicate': np.int8
}
train = pd.read_csv('train.csv', dtype=dtype)
train = pd.read_csv('train.csv', encoding='gbk')
train = pd.read_csv('file.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

### csv保存
train.to_csv('output.csv', index=False)

# JSON读取&保存
# 看需要加encoding
import json
with open('input.json') as f:
    j = json.load(f)
with open('output.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False)


Series(np.arange(3, 6), index=['a','b','c'], dtype=np.int32)

X_all = DataFrame({
    'cover_ratio': [1,2,3],
    'occur_ratio': ['a', 'b', 'c']
})
DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])

train.loc[[6, 5, 3], ['MasVnrType', 'MasVnrArea', 'Electrical']]

all_titles = pd.concat([train.title, test.title]).unique()

df.sort_values(by='ratio', ascending=False, inplace=True)
seris.sort_values(ascending=False)

train.drop(['f1', 'f2'], axis=1, inplace=True)
train.drop(train[train.score > 800].index, inplace=True)
train.drop(train[(train.score > 400) & (train.target == 1)].index, inplace=True)

num_features = train.select_dtypes(exclude = ['object']).columns
cat_features = train.select_dtypes(include = ["object"]).columns

def start_with(q, symbols):
    s = q.split()
    return int(s[0] in symbols) if len(s) > 0 else 0

train['q1_start_with_what'] = train['question1_refine'].apply(start_with, symbols=('what', "what's"))

df = df1.merge(df2, how='left', left_on='Idx', right_on='Idx')

#### Python

list_str_set = [1,2,3]
for i, v in enumerate(list_or_str):
    print(i, v)

all_chars = ['a', 'd', 'e']
char_indices = {c:i for i, c in enumerate(all_chars)}
char_dict = {i:c for i, c in enumerate(all_chars)}

#### String format
'%s %s' % ('one', 'two')
'{1} {0}'.format('one', 'two') # 反过来
'{0} {0}'.format('one') # one one这样就不用把one写2遍了

# padding
'{:10}'.format('test') # 'test      '
'{:>10}'.format('test') # '      test'
'{:^5}'.format(3) # '  3  '

'{:_^5}'.format(3) #  '__3__'

# Truncating
'{:.5}'.format('xylophone') # xylop

# padding & Truncating
'{:10.5}'.format('xylophone') # 'xylop     '

# Number
'{:d}'.format(42)
'{:f}'.format(3.141592653589793) # '3.141593' 默认使用了f，会被截短！！！
'{}'.format(3.141592653589793) # '3.141592653589793'

# #Format 最常用Float
'{:.4f}'.format(3.141592653589793) # '3.1416'
'{:6.2f}'.format(3.141592653589793) # '  3.14'
'{:06.2f}'.format(3.141592653589793) # '003.14'

# Named placeholders
'{first} {last}'.format(first='Hodor', last='Hodor!')

# datetime，不要用date
from datetime import datetime
d = datetime.strptime("2019-03-03 12:24:01", "%Y-%m-%d %H:%M:%S") # 和PHP不同
d.strftime("%Y-%m-%d %H:%M:%S")


# object
class Data(object):
    def __str__(self):
        return 'str'
    def __repr__(self):
        return 'repr'
'{0!s} {0!r}'.format(Data()) # str repr

#### Regular Expression
import re

pattern = re.compile(r'(\d+)\W(\w+)')
s = "I am 26 years old and I have 5 brothers"

# Match
#   re.match只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回None
print('\n****** Match ******')
r = re.match(r'^I', s)
if r:
    print(r.group()) # I

# Search
#   re.search 扫描整个字符串并返回**第一个**成功的匹配。
#   所以匹配不到5 brothers
print('\n****** Search ******')
r = re.search(pattern, s)
if r:
    # print(r)
    print(r.groups()) # ('26', 'years')
    print(r.group()) # 26 years # groups()返回整个匹配字符串
    print(r.group(0)) # 26 years # groups(0)返回整个匹配字符串
    print(r.group(1)) # 26 # group(1)返回第一个括号对应的匹配
    print(r.group(2)) # years

# Sub
#   替换字符串匹配项
print('\n****** Sub ******')
sub_s = re.sub(pattern, '|', s)
print(sub_s) # I am | old and I have |

# 替换函数
def sub_func(matched):
    num = int(matched.group(1)) + 1
    return '{} {}'.format(num, matched.group(2))

sub_s = re.sub(pattern, sub_func, s)
print(sub_s) # I am 27 years old and I have 6 brothers

# FindAll
#   替换字符串匹配项
print('\n****** FindAll ******')
f = re.findall(pattern, s)
print(type(f)) # <class 'list'>
print(f) # [('26', 'years'), ('5', 'brothers')]

#### Feature Engineering

def fill_nan(f, method):
    if method == 'most':
        common_value = pd.value_counts(train_master[f], ascending=False).index[0]
    elif method == 'mean:
        common_value = train_master[f].mean()
    train_master.loc[train_master[f].isnull(), f] = common_value
# 通过pd.value_counts(train_master[f])的观察得到经验
fill_nan('UserInfo_1', 'most')

submission = pd.DataFrame({
    "id": test.id,
    "result": y_pred
})
submission.to_csv('prediction.csv', index=False)

ndarray.to_csv('prediction.csv', header=True, index=True)

#### matplotlib
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
%matplotlib inline

# Figure
fig =plt.figure(num=1, figsize = (10, 10))

# Axes，单个axes
ax = fig.add_subplot(111)
ax.set(xlim=[0.5, 4.5], ylim=[-2, 8], title='An Example Axes', ylabel='Y-Axis', xlabel='X-Axis')
plt.show()

# 绘图
# 这样把个的东西绘制在一个图上！
ax.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)#绘制线
ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')#绘制散点图

为了简便，也有plt.plot，plt.title，plt.scatter，plt.xlim，plt.show。不过这是快捷用法。

# 多个Axes
fig, axes = plt.subplots(2, 2, figsize = (10, 10), sharex=True)
axes[0,0].set(...title,xlim...)
axes[0,0].imshow(…)
axes[0,0].plot(…)
axes[0,0].scatter(…)
for ax in axes.flat:
    ax.plot(…)
plt.show()

subplots直接返回fig和axes，是fig = plt.figure()，ax = fig.add_subplot(111)等等的简单写法。。

# 3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def scatter_3d(data):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(data[0], data[1], data[2], c='r', marker='o')

#### Seaborn

import seaborn as sns
sns.set(style='whitegrid')

sns.stripplot(data=train.SalePrice, jitter=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
sns.distplot(train.SalePrice, fit=norm, ax=ax1)
sns.distplot(np.log1p(train.SalePrice), fit=norm, ax=ax2)

# advanced
num_melt = pd.melt(train, id_vars=['SalePrice'], value_vars = [f for f in numerical_features])
g = sns.FacetGrid(data=num_melt, col="variable", col_wrap=4, sharex=False, sharey=False)
g.map(sns.regplot, 'value', 'SalePrice')
g.map(sns.distplot, "value")
g.map(sns.countplot, 'value', palette="muted")
g.map(sns.boxplot, 'value', 'SalePrice', palette="muted")
g.map(sns.stripplot, 'value', 'SalePrice', jitter=True, palette="muted")

corr = train.corr()
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, cmap=sns.diverging_palette(240, 10, as_cmap = True), ax=ax)

#### Random Python
import random
print('\n***** Python Random *****')
# random.seed(111)

print(random.randint(2,4)) # 2,3,4中的一个

print(random.choice([1,2,3,4])) # 选择一个
#random.choices

l = [1,2,3,4]
random.shuffle(l) # inplace
print(l) # [4, 3, 1, 2]

# random.sample

# 0-1之间的随机数
print(random.random()) # 0.5265413912197312

# start到end之间的均匀分布随机数
print(random.uniform(0, 10))

# mu=0, sigma=1的高斯分布取一个随机值
print(random.gauss(0, 1))


#### Random Numpy
import numpy as np
print('\n***** Numpy Random *****')

################Simple random data####################

# uniform distribution
print(np.random.rand(2, 3))
# [[ 0.862651    0.07646984  0.94976024]
#  [ 0.17682093  0.86376723  0.7870779 ]]

#“standard normal” distribution
print(np.random.randn(2, 3))
# [[-0.6592159  -1.25638324 -0.16564466]
#  [-0.54032012  0.13584836 -1.10408671]]

# numpy.random.randint(low, high=None, size=None, dtype='l')
print(np.random.randint(low=1, high=5, size=5)) # [1 1 4 1 2]

print(np.random.random_sample()) # 0.22454530048643417
print(np.random.random_sample(size=4)) # [ 0.62813059  0.98560066  0.65122934  0.36086048]
print(np.random.random_integers(low=1, high=5, size=3)) # [1 3 3]

# numpy.random.choice(a, size=None, replace=True, p=None)
print(np.random.choice(10, 3)) # [6 8 3]
print(np.random.choice(['a','b','c','d'], 2, p=[0.9, 0.08, 0.01, 0.01])) # ['a' 'a']

#################Permutations###################

print(np.arange(10)) # [0 1 2 3 4 5 6 7 8 9]

arr = np.arange(start=1, stop=10, step=2)
print(arr) # [1 3 5 7 9]
np.random.shuffle(arr)
print(arr) # [5 1 9 7 3]

#################Distribution###################

# numpy.random.normal(loc=0.0, scale=1.0, size=None)
print(np.random.normal(loc=0, scale=1, size=10))
# [ 0.51795205 -1.64528162 -0.6906818   0.88025762  0.69980895 -0.29976349
#   0.6081879  -1.7873237  -0.03242506  0.53256956]

print(np.random.uniform(low=0, high=1, size=10))
# [ 0.85868341  0.94486446  0.75427028  0.69226142  0.21555906  0.36668903
#   0.42261567  0.33899154  0.08086831  0.02638306]

#### cv2 opencv
# pip install opencv-python
import cv2
img = cv2.imread(path)
img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


# .astype(np.uint8)非常必须，调用cv2.cvtColor前必须转为uint8
img_pred = img_pred.astype(np.uint8)
img_pred_bgr = cv2.cvtColor(img_pred, cv2.COLOR_LAB2BGR)
img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_LAB2RGB)

# write BGR format
cv2.imwrite("a.jpg", img_pred_bgr)

# display RGB format
plt.imshow(img_pred_rgb)
plt.show()

#### plot_model

from keras.utils.vis_utils import plot_model
from IPython.display import Image, display
plot_model(model, to_file="/tmp/model.png", show_shapes=True)
display(Image('/tmp/model.png'))

#### Scikit-learn

from sklearn.model_selection import train_test_split
train, val = train_test_split(data, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size = 0.3)

from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=3, shuffle=True)

from sklearn.model_selection import cross_val_score
score = cross_val_score(LogisticRegression(), X_all, y_all, scoring='neg_mean_squared_error', cv=cv).mean()
score = cross_val_score(LogisticRegression(), X_all, y_all, scoring='accuracy', cv=cv).mean()

#### Learning Curve

from scikitplot import plotters as skplt
skplt.plot_learning_curve(LogisticRegression(), X_all, y_all)
plt.show()
skplt.plot_roc_curve(y_true=y_val, y_probas=y_proba)
plt.show()
skplt.plot_precision_recall_curve(y_true=y_val, y_probas=y_proba)
plt.show()
skplt.plot_confusion_matrix(y_true=y_val, y_pred=y_pred, normalize=True)
plt.show()

#### XGBoost

from xgboost import XGBRegressor
import xgboost as xgb
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
}
dtrain = xgb.DMatrix(X_all, label=y_all)
history = xgb.cv(params, dtrain, num_boost_round=1024, early_stopping_rounds=5, verbose_eval=20)

booster = xgb.train(params, dtrain)
xgb.plot_importance(booster=booster)
xgb.plot_tree(booster=booster)
xgb.to_graphviz(booster=booster)

#### StandardScaler

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_all[:] = std.fit_transform(X_all)

#### GridSearchCV

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [60, 70, 80, 90],
    'learning_rate': [2, 3, 4, 5]
}
gs = GridSearchCV(
    AdaBoostRegressor(),
    param_grid = param_grid
)
gs.fit(X_all, y_all)
gs.best_params_

#### Keras

from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, UpSampling2D, Lambda
from keras.models import Model, Input
from keras.callbacks import Callback, CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
import keras.backend as K

def my_loss():
    cross_entropy = K.categorical_crossentropy(y_pred, y_true)
    cross_entropy = K.mean(cross_entropy, axis=-1)
    return cross_entropy

def get_model():
    img_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    x = img_input
    print(x.get_shape())

    x = Dense(32, activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

    model.add(LSTM(LSTM_UNITS, input_shape=(SEN_LEN, len(all_chars)), return_sequences=True))
    model.add(LSTM(LSTM_UNITS, input_shape=(SEN_LEN, len(all_chars)), return_sequences=False))

    # 新的写法
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))

    # 废弃的写法
    # gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal')(x)
    # gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(x)
    # # keras.layers.add
    # gru1_merged = add([gru_1, gru_1b])

    x = K.reshape(x, (-1, AB_PAIRS))
    x = K.softmax(x)
    x = K.resize_images(x, IMG_HEIGHT / curr_height, IMG_WIDTH / curr_width, data_format="channels_last")

    x = Lambda(lambda z: reshape(z), output_shape=(IMG_HEIGHT, IMG_WIDTH, AB_PAIRS))(x)

    model = Model(inputs=img_input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

model.load_weights('/input/Captcha/checkpoint-33-0.3923.hdf5')

class MyEvaluator(Callback):
    def __init__(self):
        self.gen = gen(1, random_choose=True)
    
    # self.model integrated
    def on_epoch_end(self, epoch, logs=None):
        X_test, y_test = next(self.gen) # (1, 128, 128, 1)
        y_pred = self.model.predict(X_test) # (1, 128, 128, AB_PAIRS)
evaluator = MyEvaluator()


model = get_model()
print('model built')

RUN = RUN + 1 if 'RUN' in locals() else 1
print("RUN {}".format(RUN))

LOG_DIR = '/output/training_logs/run-{}'.format(RUN)
LOG_FILE_PATH = LOG_DIR + '/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5'

csv_log = callbacks.CSVLogger(filename =  LOG_DIR + '/log.csv')
tensorboard = TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_grads=False, write_graph=False)
checkpoint = ModelCheckpoint(filepath=LOG_FILE_PATH, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: learning_rate * (0.9 ** epoch))
# schedule: a function that takes an epoch index as input (integer, indexed from 0)
# and returns a new learning rate as output (float).

# verbose也可以是2，从而不输出进度条，每个epoch只输出一个val结果
m1.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.3,
                  callbacks=[tensorboard, checkpoint, early_stopping], verbose=1)

history = model.fit_generator(generator=gen(8), steps_per_epoch=64,
                              validation_data=gen(8, random_choose=True), validation_steps=8,
                              epochs=10000, verbose=1,
                              callbacks=[csv_log, tensorboard, checkpoint, early_stopping, lr_decay, evaluator])

#### ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(data_format='channels_last').flow_from_directory(
    './train/', target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True)

keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=K.image_data_format())

#### TensorFlow

# refer to: 
# RecommendSystem.ipynb
# DigitRecognizer_tensorflow.ipynb

import tensorflow as tf

# placeholder & Variable
a = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, IMAGE_PIXELS), name='input_images')
b = tf.constant(train_matrix, dtype=tf.float32, name='const_name')
b_bool = tf.cast(tf_train_matrix, dtype=tf.bool, name='cast_name')
c = tf.Variable(train_matrix, dtype=tf.float32, name='v_name')

# calculation
with tf.name_scope('hidden1'):
  pred_matrix = tf.tensordot(tf_user_features, tf_item_features, axes=[[1], [1]])
  pred_matrix_int = tf.cast(tf.round(pred_matrix), tf.int32)
  train_squared_diff_filtered = tf.where(tf_train_matrix_bool, tf.square(train_diff), tf_matrix_mean)

# loss definition and scalar record
loss = 0.1 * tf.sqrt(tf.reduce_sum(train_squared_diff_filtered))
tf.summary.scalar('loss', loss)

# Evaluation
evaluation_op = tf.reduce_sum(tf.where(tf_val_matrix_bool, tf.abs(val_diff), tf_matrix_zero_float)) / 20000
tf.summary.scalar('evaluation_value', evaluation_op)

# train operation
train_op = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(loss)

# summary
summary_op = tf.summary.merge_all()

# session
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  # saver & summary_writer
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter("./log2/", sess.graph)

  for step in range(102400):
      feed_dict = {
          'placeholder1': <value>,
          'placeholder2': <value>,
      }
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

      if (step + 1) % 128 == 0:
          # 其实scalar里面已经记录了，这里可以认为是一个自定义evaluator
          evaluation_score = sess.run(evaluation_op, feed_dict=feed_dict)
          saver.save(sess, "./log/cp-file.ckpt", global_step=step)

      if some-cond:
          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          summary_writer.flush()

#### keras tensorflow
# ref to StyleTransfer.ipynb

#### Embedding
# ref to: MNIST/Embeddings.ipynb

只要存在Embeddings，Keras会自动记录。TB里会直接看到。我只需要手动生成meta-data文件，不管是程序中绑定，还是在TB中Load都可以。至于meta-data文件，提前生成也OK，程序生成也OK。

#### NPL, NLTK, ngrams
# ref to: Quora/Quora-neural-network.ipynb
# ref to: Quora/Quora-Feature-Enginnering.ipynb

#### word2vec
from gensim.models.word2vec import Word2Vec, KeyedVectors
word2vec = KeyedVectors.load_word2vec_format('/input/Kaggle/Word2Vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
'apple' in word2vec.vocab # True

#### Tokenizer-Scikit-Learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import EnglishStemmer

# 自定义analyzer，加入stem
count_analyzer = CountVectorizer().build_analyzer()
stemmer = EnglishStemmer()

def stem_count_analyzer(doc):
    return (stemmer.stem(w) for w in count_analyzer(doc))

cv = CountVectorizer(analyzer=stem_count_analyzer, preprocessor=None, stop_words='english', max_features=128)
cv.fit(unique_questions)
q1_cv = cv.transform(train.question1)

# 使用默认word analyzer。再加入preprocessor
def preprocessor(review):
    return BeautifulSoup(review, 'html5lib').get_text()
count_v = CountVectorizer(analyzer='word', preprocessor=preprocessor, stop_words='english', max_features=5000)

#### Keras Tokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(all_unique_questions)

q1_seq = tokenizer.texts_to_sequences(train.question1)
q1_seq_pad = pad_sequences(q1_seq, maxlen=MAX_SEQ_LEN)

input_size = len(tokenizer.word_index) + 1 # 0 index is all 0
shared_embedding = Embedding(input_dim=input_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LEN,
                                weights=[embedding_weights], trainable=False, name='shared_embedding_layer')

# N-grams

from nltk.util import ngrams
ngrams([1,2,3,4,5], 2)

#### KMeans, MiniBatchKMeans, DBSCAN
# ref to: Iris/algorithm.ipynb

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN

kmeans = KMeans(n_clusters=5000, max_iter=10, n_jobs=8)
kmeans.fit(X_all)
X_all_km_centroids = [kmeans.cluster_centers_[idx] for idx in kmeans.labels_]

minibatchkmeans = MiniBatchKMeans(n_clusters=5000, max_iter=10)
minibatchkmeans.fit(X_all)
X_all_mbkm_centroids = [minibatchkmeans.cluster_centers_[idx] for idx in minibatchkmeans.labels_]

dbscan = DBSCAN()
dbscan.fit(X_all)
len(X_all), len(dbscan.core_sample_indices_), len(dbscan.components_), len(dbscan.labels_)

#### get_file

from keras.utils import get_file
path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
with open(path, 'r') as f:
  data = f.read()

#### tqdm

from tqdm import tqdm
for i in tqdm(range(100)):
  pass

#### glob

from glob import glob
file_glob = glob('{}/*.jpg'.format(DATA_DIR))

#### Thread

from threading import Thread

def handle_img(**kwargs):
    i = kwargs['i']

threads = []
for i in range(batch_size):
    kwargs = {
        'i': i,
    }
    t = Thread(target=handle_img, kwargs=kwargs)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

#### Logging

import logging
logger = logging.getLogger('log')
logger.setLevel(logging.WARNING)
logger.debug('abc')
logger.info('abc')
logger.warning('abc')
logger.error('abc')
logger.critical('abc')
logger.log('abc')
logger.exception('abc')

#### BeautifulSoup

from bs4 import BeautifulSoup
BeautifulSoup("<p>hello,workd</p>", 'html5lib').get_text()

#### time

from time import sleep
sleep(.5)

#### Arrow

import arrow

def parse_date(date_str, str_format='YYYY/MM/DD'):
    d = arrow.get(date_str, str_format)
    # 月初，月中，月末
    month_stage = int((d.day-1) / 10) + 1
    return (d.timestamp, d.year, d.month, d.day, d.week, d.isoweekday(), month_stage)

#### Hypertools
# ref to: Iris/visualization.ipynb

from sklearn import datasets
import hypertools as hyp

digits = datasets.load_digits(n_class=4)
data = digits.data
group = digits.target.astype('str')

hyp.plot(data, '.', group=group, legend=list(set(group)))
# hyp.plot(data, '.', group=group, legend=list(set(group)), model='TSNE') # 慢了很多

# hyp.plot(data, '.', n_clusters=4, ndims=2)


