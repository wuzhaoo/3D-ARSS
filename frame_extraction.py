import os
import random
def frame_extraction():
    result=[]
    res=0
    for i in range(11):
        DATA_PATH = "./dataset/sequences/%02d/velodyne" %i
        all_data_name = os.listdir(DATA_PATH)
        all_data_name.sort(key=lambda x:int(x[:-4]))

        all_data_name = all_data_name[1:-1]

        print(('sequence/%02d:\t'%i)+str(len(all_data_name)))
        res=res+len(all_data_name)
        j=0
        while j <(len(all_data_name)) :
            all_data_name[j]=DATA_PATH+'/'+all_data_name[j]
            result.append(all_data_name[j])
            j+=10
    print('total',    (res))

    print('result',len(result))
    # return result
print(frame_extraction())
