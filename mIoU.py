import torch
import torch.nn as nn
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
class SegmentationMetric(object):
    def __init__(self,numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
    def mIoU(self):
        intersction = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix,axis=1)+np.sum(self.confusionMatrix,axis=0)-np.diag(self.confusionMatrix)
        IoU=intersction/union
        # IoU=intersction[1:]/union[1:]

        mIoU=np.nanmean(IoU)
        return mIoU
    def precision(self):
        precision = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1)+1e-15)
        # precision = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return precision  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率


    def recall(self):
        recall = np.diag(self.confusionMatrix)/ ((self.confusionMatrix.sum(axis=0))+1e-15)
        # recall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return recall
    def F1Score(self):
        return  2*self.recall()*self.precision()/(self.recall()+self.precision())
    def addBatch(self,Predict,Label):
        Predict=Predict.squeeze()
        Label=Label.squeeze()
        assert Predict.shape==Label.shape
        mask = (Label >= 0) & (Label < self.numClass)
        label = self.numClass * Label[mask] + Predict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        self.confusionMatrix+=confusionMatrix
    def reset(self):
        self.confusionMatrix=np.zeros((self.numClass,)*2)

if __name__ == '__main__':

    pred=np.array([0,1,1,1,1,1,2,3,4,5,6,6])
    target=np.array([0,0,0,0,1,1,2,3,4,5,5,6])
    metric=SegmentationMetric(7)
    metric.addBatch(pred,target)

    print("mIoU=",metric.mIoU())
    print("precision=",metric.precision())
    print("recall=",metric.recall())
    print("F1Score=",metric.F1Score())


