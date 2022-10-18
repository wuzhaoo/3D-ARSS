import os
import random
import numpy as np
import sys
a=[]
result = []
train_size=0.6
validate_size=0.2
test_size=0.2
MAX_Sequence_Length=466*2
for i in range(11):
    res=[]
    DATA_PATH = "./dataset/sequences/%02d/velodyne" %i
    all_data_name = os.listdir(DATA_PATH)
    all_data_name.sort(key=lambda x:int(x[:-4]))
    all_data_name = all_data_name[1:-1]
    j=0
    all_data_name_select=[]
    while j < (len(all_data_name)):
        all_data_name_select.append(DATA_PATH + '/' + all_data_name[j])
        j+=5#####per frame every 5frame
    random.seed(0)
    random.shuffle(all_data_name_select)
    train_ = all_data_name_select[:int(len(all_data_name_select) * train_size)]
    validate_ = all_data_name_select[int(len(all_data_name_select) * 0.6):int(len(all_data_name_select) * 0.8)]
    test_ = all_data_name_select[int(len(all_data_name_select) * 0.8):int(len(all_data_name_select))]

    print('len(all_data_name)',len(all_data_name_select))
    while len(train_)<5000:
        train_+=train_
        validate_ += validate_
        test_+=test_

    #every sequence has same length of frame
    result+=train_[0:int(MAX_Sequence_Length*0.6)]
    # result+=validate_[0:int(MAX_Sequence_Length* 0.2)]
    # result+=test_[0:int(MAX_Sequence_Length* 0.2)]

# print(len(result))
# for i in result:
#     print(i)

result=np.asarray(result)
result=result.reshape(11,-1)
result_T=result.T
result_T=result_T.reshape(1,-1).squeeze()
np.set_printoptions(threshold=sys.maxsize)
with open("config_train.txt", "w") as f:
    for i in result_T:
        f.write(str(i)+'\n')
