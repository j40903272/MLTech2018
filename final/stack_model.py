import numpy as np
import pandas as pd

class FM():
    
    def __init__(self, fm_ttt):
        from sklearn.feature_extraction import DictVectorizer
        self.feature_list = ['User-ID', 'ISBN', 'loc']
        tmp = self.trans(fm_ttt)
        self.v = DictVectorizer()
        self.v.fit(tmp)
        
    def build(self):
        from pyfm import pylibfm
        self.model = pylibfm.FM(num_factors=5, num_iter=4, verbose=False, task="regression")
    
    def fit(self, X, Y, X_val, Y_val):
        self.build()
        tmp = self.trans(X)
        y = Y.values.astype(float)
        x = self.v.transform(tmp)
        self.model.fit(x, y)
    
    def predict(self, X):
        tmp = self.trans(X)
        x = self.v.transform(tmp)
        pred = self.model.predict(x)
        return np.array(pred)
    
    def trans(self, data):
        dic = data[self.feature_list].to_dict('index')
        tmp = [dic[i] for i in dic]
        return tmp



class Surprise():
    
    def build(self):
        from surprise import SVD
        self.model = SVD(
        n_epochs=70,
        n_factors=0,
        lr_bi=0.002,
        reg_all=0.0,
        use_l1_loss=True)
        
    def fit(self, X, Y, X_val, Y_val):
        self.build()
        from surprise import Dataset, Reader
        x = X[['User-ID', 'ISBN']].copy()
        x['Book-Rating'] = Y.values
        train = Dataset.load_from_df(x, Reader(rating_scale=(1, 10)))
        self.model.fit(train.build_full_trainset())

    def predict(self, X):
        pred = []
        for u, b in zip(X['User-ID'], X['ISBN']):
            pred.append(self.model.predict(u, b)[3])
        return np.array(pred)

    

class XGBR():
    
    def build(self):
        from xgboost import XGBRegressor
        self.model = XGBRegressor(n_estimators=2000, learning_rate =0.01, max_depth=22, min_child_weight=11, gamma=0.2,
                                  colsample_bytree=0.6, subsample=0.9, reg_alpha=0.1, reg_lambda=1e-05)
    
    def fit(self, X, Y, X_val=None, Y_val=None):
        self.build()
        if type(X_val) == type(None):
            self.model.fit(X, Y, verbose=False)
        else:
            self.model.fit(X, Y, early_stopping_rounds=10, eval_set=[(X_val, Y_val)], verbose=False)
    
    def predict(self, X):
        return self.model.predict(X)

    

class XGBC():
    
    def build(self):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(n_estimators=1000, max_depth=10)
    
    def fit(self, X, Y, X_val=None, Y_val=None):
        self.build()
        if type(X_val) == type(None):
            self.model.fit(X, Y, verbose=False)
        else:
            self.model.fit(X, Y, early_stopping_rounds=10, eval_set=[(X_val, Y_val)], verbose=False)
    
    def predict(self, X):
        return self.model.predict(X)


class Keras():
    
    def __init__(self):
        self.set_gpu_memory()
                         
    def fit(self, X, Y, X_val, Y_val):
        self.build()
        X, Y = [X['User-ID'], X['ISBN']], Y.values
        self.model.fit(X, Y, epochs=25, verbose=False)
    
    def predict(self, X):
        X = [X['User-ID'], X['ISBN']]
        return self.model.predict(X)
    
    def build(self):
        ttt = pd.read_csv('merge.csv')
        n_users, n_books = ttt['User-ID'].nunique(), ttt['ISBN'].nunique()
        n_latent_factors = 3
        
        from keras import Model
        from keras.layers import Dense, Dropout, Input, Flatten, Embedding, merge
                         
        book_input = Input(shape=[1], name='Item')
        book_embedding = Embedding(n_books + 1, n_latent_factors, name='Book-Embedding')(book_input)
        book_vec = Flatten(name='FlattenBooks')(book_embedding)

        user_input = Input(shape=[1],name='User')
        user_vec = Flatten(name='FlattenUsers')(Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input))

        input_vecs = merge([book_vec, user_vec], mode='concat', name='concat')
        y = Dense(1)(input_vecs)
                         
        model = Model([user_input, book_input], y)
        model.compile('sgd', 'mae')
        
        self.model = model
    
    def set_gpu_memory(self):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        set_session(tf.Session(config=config))
        
        
class nn():
    def __init__(self):
        ttt = pd.read_csv('merge.csv')
        self.n_users = ttt['User-ID'].nunique()
        self.n_books = ttt['ISBN'].nunique()
    
    def build(self, global_mean):
        from nn.model import Model
        self.model = Model(
            user_size = self.n_users + 1,
            book_size = self.n_books + 1,
            global_mean=global_mean).cuda()
    
    def fit(self, X, Y, X_val, Y_val):
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        import torch.nn.functional as F
        from nn.model import Model
        
        user_train, book_train, rate_train = X['User-ID'].values, X['ISBN'].values, Y.values
        
        train_loader = DataLoader(
            dataset=TensorDataset(
                torch.from_numpy(user_train),
                torch.from_numpy(book_train),
                torch.from_numpy(rate_train)),
            batch_size=1024,
            shuffle=True,
            num_workers=8)
        
        global_mean = rate_train.mean().item()
        self.build(global_mean)
        optimizer_u = torch.optim.SGD(self.model.user_bias.parameters(), lr=0.01)
        optimizer_b = torch.optim.SGD(self.model.book_bias.parameters(), lr=0.005)
        
        for epoch in range(45):
            for i ,(user, book, rate) in enumerate(train_loader):
                self.model.train()
                user, book, rate = user.cuda(), book.cuda(), rate.cuda()
                output, reg_loss = self.model(user.long(), book.long())

                # MAE Loss
                loss = F.l1_loss(output, rate.float(), size_average=False)
                # MAPE Loss
                #loss = (F.l1_loss(output, rate, reduce=False) / rate).sum()

                optimizer_u.zero_grad()
                optimizer_b.zero_grad()
                loss.backward()
                optimizer_u.step()
                optimizer_b.step()
    
    def predict(self, X):
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        
        self.model.eval()
        user_test, book_test = X['User-ID'].values, X['ISBN'].values
        test_loader = DataLoader(
            dataset=TensorDataset(
                torch.from_numpy(user_test),
                torch.from_numpy(book_test)),
            batch_size=1024,
            num_workers=8)
        pred = []
        for i ,(user, book) in enumerate(test_loader):
            user, book= user.cuda(), book.cuda()
            output = self.model(user.long(), book.long(), test=True)
            pred.append(output.detach().cpu().numpy())

        pred = np.hstack(pred)
        return np.round(pred)
