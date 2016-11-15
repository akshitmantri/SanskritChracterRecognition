require 'paths'
require 'nn'
require 'xlua'

trainset = torch.load('Dataset/trainset.t7')
testset = torch.load('Dataset/testset.t7')

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.labels[i]} 
                end}
);

trainset.data = trainset.data:double()

function trainset:size() 
    return self.data:size(1) 
end

testset.data = testset.data:double()

function testset:size() 
    return self.data:size(1) 
end

mean = trainset.data:mean()
print ('Mean = ' .. mean)
trainset.data = trainset.data:add(-mean)
stdv = trainset.data:std()
print('Stdv = ' .. stdv)
trainset.data = trainset.data:div(stdv)

testset.data = testset.data:double()
testset.data = testset.data:add(-mean)
testset.data = testset.data:div(stdv)

net = torch.load('testmodel1.t7')
-- net = nn.Sequential()
-- net:add(nn.SpatialConvolution(1, 6, 5, 5)) 
-- net:add(nn.ReLU())                       
-- net:add(nn.SpatialMaxPooling(2,2,2,2))     
-- net:add(nn.SpatialConvolution(6, 16, 5, 5))
-- net:add(nn.ReLU())                       
-- net:add(nn.SpatialMaxPooling(2,2,2,2))
-- net:add(nn.View(16*5*5))                 
-- net:add(nn.Linear(16*5*5, 120))          
-- net:add(nn.ReLU())                       
-- net:add(nn.Linear(120, 84))
-- net:add(nn.ReLU())                       
-- net:add(nn.Linear(84, 10))               
-- net:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

function correct_12(dataset,net,correct)
    -- body
    for i=1,dataset:size() do
        local groundtruth = dataset.labels[i][1]
        local prediction = net:forward(dataset.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            correct = correct + 1
        end
        xlua.progress(i,dataset:size())
    end
    print(correct, 100*correct/dataset:size() .. ' % ')
end

correct_12(testset,net,0)   
for l=1,5 do  
    currentError = 0
    currentLearningRate = 0.001
    for t = 1,trainset:size() do
        example = trainset[t];
        input = example[1];
        target = example[2];
        error = criterion:forward(net:forward(input), target);
        currentError = currentError + error
        net:updateGradInput(input, criterion:updateGradInput(net.output, target));
        net:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate);
        xlua.progress(t,trainset:size())
    end
    currentError = currentError / trainset:size()
    print("# current error = " .. currentError)
    correct_12(testset,net,0)
    torch.save('testmodel1.t7',net)
    print("model saved")
end



