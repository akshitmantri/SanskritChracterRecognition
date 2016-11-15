matio = require 'matio'

trainlabel = matio.load('Dataset/train_label_hog.mat')
t1 = matio.load('Dataset/train_data_hog.mat')
t2 = matio.load('Dataset/test_data_hog.mat')
testlabel=  matio.load('Dataset/test_label_hog.mat')

trainset = t1.train_data
testset = t2.test_data
trainlabel = trainlabel.train_label
testlabel = testlabel.test_label

trainset = {
	data = trainset,
	labels = trainlabel
}

testset = {
	data = testset,
	labels = testlabel
}

function trainset:size() 
    return self.data:size(1) 
end

function testset:size() 
    return self.data:size(1) 
end

trainDataTemp = trainset
testDataTemp = testset


shuffle = torch.randperm(trainset:size())
for i = 1,trainset:size() do
    trainset.data[i] = trainDataTemp.data[shuffle[i]]
    trainset.labels[i] = trainDataTemp.labels[shuffle[i]]
end
shuffle = torch.randperm(testset:size())
for i = 1,testset:size() do
    trainset.data[i] = testDataTemp.data[shuffle[i]]
    trainset.labels[i] = testDataTemp.labels[shuffle[i]]    
end

torch.save('trainset.t7',trainset)
torch.save('testset.t7',testset)
