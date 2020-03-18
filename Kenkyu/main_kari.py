import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable
import math
import numpy as np
import pandas_datareader.data as web
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import time



"""
    params\n
    data_size : データセットサイズ\n
    data_length : 各データの時系列長\n
    freq : 周波数\n
    noise : ノイズの振幅\n
    returns\n
    train_x : トレーニングデータ（t=1,2,...,size-1の値)\n
    train_t : トレーニングデータのラベル（t=sizeの値）\n
    
"""

class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        
        self.drop = nn.Dropout(p = 0.5)
        
        self.output_layer = nn.Linear(hiddenDim, outputDim)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, inputs, hidden0=None):

        output, (hidden, cell) = self.rnn(inputs, hidden0)
        
        output = self.drop(output)
        #print("outputttttttttttttttttt",output)
        #print("#####################################",output[:, -1, :])

        output = self.output_layer(output[:, -1, :])

        #output = self.output_layer(output)
        
        output = self.sig(output)
        
        return(output)

def mkDataSet(start, end, input_len):
  
    train_x = []
    train_t = []
    
    data = web.DataReader(["DEXJPUS","DEXUSEU","DEXCHUS"], "fred", start, end)
    data = data.interpolate()

    data_list = data.values.tolist()

    #　３つの移動平均を準備
    #　ih1_len　　移動平均をとる長さ、それぞれ短期中期長期
    #　当日の値を重視した移動平均を採用
    
    ih1_len = 5
    ih2_len = 20
    ih3_len = 60
    
    ih1 = []
    ih2 = []
    ih3 = []
    goukei = 0
    
    # それぞれの移動平均の配列を0で埋める
    for i in range(ih1_len):
      ih1.append(0)
    for i in range(ih2_len):
      ih2.append(0)
    for i in range(ih3_len):
      ih3.append(0)
    

    #　1日目を計算、当日だけ2度加算し、長さ＋１で割る
    for i in range(ih1_len):
      goukei = goukei + (data_list[i][0])/(ih1_len + 1)
      if(i == ih1_len-1):
        goukei = goukei + (data_list[i][0])/(ih1_len + 1)
    ih1[ih1_len-1] = goukei
    goukei = 0
    for i in range(ih2_len):
      goukei = goukei + (data_list[i][0])/(ih2_len + 1)
      if(i == ih2_len-1):
        goukei = goukei + (data_list[i][0])/(ih2_len + 1)
    ih2[ih2_len-1] = goukei
    goukei = 0
    for i in range(ih3_len):
      goukei = goukei + (data_list[i][0])/(ih3_len + 1)
      if(i == ih3_len-1):
        goukei = goukei + (data_list[i][0])/(ih3_len + 1)
    ih3[ih3_len-1] = goukei
    
    # 二日目以降
    # i-1 日目までの平均　＋　（i日目の2倍　－　i-1日目　－　最も古い日）/　（長さ＋１）
    for i in range(len(data_list)):
      if(i > ih1_len-1):
        ih1.append(ih1[i-1] + (2*data_list[i][0] - data_list[i-1][0] - data_list[i-ih1_len][0]) / (ih1_len + 1))
      if(i > ih2_len-1):
        ih2.append(ih2[i-1] + (2*data_list[i][0] - data_list[i-1][0] - data_list[i-ih2_len][0]) / (ih2_len + 1))
      if(i > ih3_len-1):
        ih3.append(ih3[i-1] + (2*data_list[i][0] - data_list[i-1][0] - data_list[i-ih3_len][0]) / (ih3_len + 1))
        
    # 3つの移動平均を各日ごとに
    
    for i in range(len(data_list)):
      data_list[i].append(ih1[i])
      data_list[i].append(ih2[i])      
      data_list[i].append(ih3[i])
    
    # データが存在しない分を削除、最も長い移動平均に合わせる
    del data_list[0:ih3_len]
    
    tmp= [[0] * len(data_list[0]) for i in [1] * len(data_list)]
    
    #　logで当日と前日の差分を取る、わかりやすさのため100倍、小数点以下ｎ桁までを指定
    for i in range(len(data_list)-1):
      for j in range(len(data_list[0])):
        tmp[i][j] = round(( math.log(data_list[i+1][j]) - math.log(data_list[i][j]) )*100,4)

    data_list = tmp
    
    # inputの長さずつに
    for i in range(len(data_list)-input_len-1):
      train_x.append(data_list[i : i+input_len])
    
    #　ラベル付け
      if(data_list[i+input_len][0] > 0):
        train_t.append([1])
      else:
        train_t.append([0])


    #for i in range(len(train_t)):
      #print(train_x[i])
      #print(train_t[i])
        
    
    #　データを分割
    from sklearn.model_selection import train_test_split
    (train_x, test_x,
     train_t, test_t) = train_test_split(
        train_x, train_t, test_size=0.2
    )
    
    return (train_x, train_t, test_x, test_t)

def mkRandomBatch(train_x, train_t, batch_size):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    batch_x = []
    batch_t = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x))
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
        
    return torch.Tensor(batch_x), torch.Tensor(batch_t)

def mkTestModelset(input_len):
  if(1):
    start = datetime.datetime(2017, 1, 8)
    end = datetime.datetime(2018, 12, 31)

    train_x = []
    train_t = []
    
    data = web.DataReader(["DEXJPUS","DEXUSEU","DEXCHUS"], "fred", start, end)
    data = data.interpolate()

    data_list = data.values.tolist()

    ih1_len = 5
    ih2_len = 20
    ih3_len = 60
    
    ih1 = []
    ih2 = []
    ih3 = []
    goukei = 0
    
    for i in range(ih1_len):
      ih1.append(0)
    for i in range(ih2_len):
      ih2.append(0)
    for i in range(ih3_len):
      ih3.append(0)
  
    for i in range(ih1_len):
      goukei = goukei + (data_list[i][0])/(ih1_len + 1)
      if(i == ih1_len-1):
        goukei = goukei + (data_list[i][0])/(ih1_len + 1)
    ih1[ih1_len-1] = goukei
    goukei = 0
    for i in range(ih2_len):
      goukei = goukei + (data_list[i][0])/(ih2_len + 1)
      if(i == ih2_len-1):
        goukei = goukei + (data_list[i][0])/(ih2_len + 1)
    ih2[ih2_len-1] = goukei
    goukei = 0
    for i in range(ih3_len):
      goukei = goukei + (data_list[i][0])/(ih3_len + 1)
      if(i == ih3_len-1):
        goukei = goukei + (data_list[i][0])/(ih3_len + 1)
    ih3[ih3_len-1] = goukei
    
    # 二日目以降
    # i-1 日目までの平均　＋　（i日目の2倍　－　i-1日目　－　最も古い日）/　（長さ＋１）
    for i in range(len(data_list)):
      if(i > ih1_len-1):
        ih1.append(ih1[i-1] + (2*data_list[i][0] - data_list[i-1][0] - data_list[i-ih1_len][0]) / (ih1_len + 1))
      if(i > ih2_len-1):
        ih2.append(ih2[i-1] + (2*data_list[i][0] - data_list[i-1][0] - data_list[i-ih2_len][0]) / (ih2_len + 1))
      if(i > ih3_len-1):
        ih3.append(ih3[i-1] + (2*data_list[i][0] - data_list[i-1][0] - data_list[i-ih3_len][0]) / (ih3_len + 1))
        
    # 3つの移動平均を各日ごとに
    
    for i in range(len(data_list)):
      data_list[i].append(ih1[i])
      data_list[i].append(ih2[i])      
      data_list[i].append(ih3[i])
    
    # データが存在しない分を削除、最も長い移動平均に合わせる
    del data_list[0:ih3_len]
    
    tmp= [[0] * len(data_list[0]) for i in [1] * len(data_list)]
    
    #　logで当日と前日の差分を取る、わかりやすさのため100倍、小数点以下ｎ桁までを指定
    for i in range(len(data_list)-1):
      for j in range(len(data_list[0])):
        tmp[i][j] = round(( math.log(data_list[i+1][j]) - math.log(data_list[i][j]) )*100,4)

    data_list = tmp
    
    # inputの長さずつに
    for i in range(len(data_list)-input_len-1):
      train_x.append(data_list[i : i+input_len])
    
    #　ラベル付け
      if(data_list[i+input_len][0] > 0):
        train_t.append([1])
      else:
        train_t.append([0])

  return(train_x,train_t)
  

def main(input_len, epochs_num, hidden_size, batch_size, output_size, lr):

    start = datetime.datetime(1999, 1, 8)
    end = datetime.datetime(2016, 12, 31)
    
    #test_start = datetime.datetime(2015, 1, 8)
    #test_end = datetime.datetime(2016, 12, 31)
    
    training_size = 0
    test_size = 0
    
    train_x, train_t, test_x, test_t = mkDataSet(start, end, input_len)    
    
    model = Predictor(6, hidden_size, output_size)

    test_x = torch.Tensor(test_x)
    test_t = torch.Tensor(test_t)

    train_x = torch.Tensor(train_x)
    train_t = torch.Tensor(train_t)
    #print(test_x.size())
    #print(test_t.size())

    #print(test_x)
    #print(test_t)
    #exit

    #test_x = torch.Tensor(test_x)
    #test_t = torch.Tensor(test_t)
    
    dataset = TensorDataset(train_x, train_t)
    loader_train = DataLoader(dataset, batch_size = batch_size, shuffle=True)

    dataset = TensorDataset(test_x, test_t)
    loader_test = DataLoader(dataset, batch_size = batch_size, shuffle=False)
    
    #dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=4, shuffle=True,num_workers=2)


    #torch.backends.cudnn.benchmark=True
    
    optimizer = SGD(model.parameters(), lr)

    criterion = torch.nn.BCELoss(size_average=False)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs_num * 0.3, epochs_num * 0.7], gamma=0.1, last_epoch=-1)

    loss_record = []
    count= 0
    
    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        training_num = 0
        #scheduler.step()
        model.train()

        for i, data in enumerate(loader_train, 0):
          #入力データ・ラベルに分割
          # get the inputs
          inputs, labels = data

          # optimizerの初期化
          # zero the parameter gradients
          optimizer.zero_grad()

          #一連の流れ
          # forward + backward + optimize
          outputs = model(inputs)

          labels = labels.float()

          #ここでラベルデータに対するCross-Entropyがとられる
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # ロスの表示
          # print statistics
          #running_loss += loss.data[0]
          
          running_loss += loss.data.item()*100
          
          training_accuracy += np.sum(np.abs((outputs.data - labels.data).numpy()) <= 0.5)
          
          training_num += np.sum(np.abs((outputs.data - labels.data).numpy()) != 10000)

        #test
        test_accuracy = 0.0
        test_num = 0
        model.eval()

        for i, data in enumerate(loader_test,0):
            inputs, labels = data

            outputs = model(inputs)
            
            labels = labels.float()
            #print("#######################")
            #print(outputs)
            #print(labels)

            #print(output.t_(),label.t_())
            #print(np.abs((output.data - label.data).numpy()))



            test_accuracy += np.sum(np.abs((outputs.data - labels.data).numpy()) <= 0.5)
            test_num += np.sum(np.abs((outputs.data - labels.data).numpy()) != 100000)
            
        training_accuracy /= training_num
        test_accuracy /= test_num

        
        if((epoch+1) % 1 == 0):
          print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
              epoch + 1, running_loss, training_accuracy, test_accuracy))
          #print(output)

        
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'jikkenpath')
            
        
        #loss_record.append(running_loss)
    else:
      print(training_num)
      print(test_num) 
    #print(loss_record)
    
    if(1):
      test_x, test_t = mkTestModelset(input_len)
      test_x= torch.Tensor(test_x)
      test_t= torch.Tensor(test_t)
      dataset = TensorDataset(test_x,test_t)
      loader_test = DataLoader(dataset, batch_size = 5, shuffle=False)
      test_accuracy = 0.0
      test_num = 0
      av_testac = 0
      model.eval()
      
      for i, data in enumerate(loader_test,0):
          inputs, labels = data

          outputs = model(inputs)
          
          labels = labels.float()

          test_accuracy = 0.0
          test_num = 0
          test_accuracy += np.sum(np.abs((outputs.data - labels.data).numpy()) <= 0.5)
          test_num += np.sum(np.abs((outputs.data - labels.data).numpy()) != 100000)
          
          test_accuracy /= test_num
          av_testac += test_accuracy
          
          print(i,test_accuracy)
      else: 
        print(av_testac/(i+1))

    torch.save(model.state_dict(), 'weight.pth')

if __name__ == '__main__':

  input_len = 96
  epochs_num = 400
  hidden_size = 400
  batch_size = 16
  output_size = 1
  lr = 0.001

  if(0):
    for i in range (10):
      for i in range (1):
        start = time.time()

        main(input_len, epochs_num, hidden_size, batch_size, output_size, lr)

        eltime = time.time() - start

        print ("elapsed_time:{0}".format(eltime) + "[sec]")
        print(input_len, epochs_num, hidden_size, batch_size, output_size, lr)
      input_len -= 20
    hidden_size = hidden_size/2

  else:
    start = time.time()

    main(input_len, epochs_num, hidden_size, batch_size, output_size, lr)

    eltime = time.time() - start

    print ("elapsed_time:{0}".format(eltime) + "[sec]")
    print(input_len, epochs_num, hidden_size, batch_size, output_size, lr)




