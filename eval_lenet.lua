require 'nn'
require 'image'
require 'paths'
cjson = require 'cjson'

model = torch.load('testmodel1.t7')

mean = 234.55973184915	
stdv = 46.101516773469

function rgb2gray(img)
	r = img:select(1,1):double()
	g = img:select(1,2):double()
	b = img:select(1,3):double()
	z = torch.Tensor(img:size(2),img:size(3)):zero()
	z = z:add(0.21,r)
	z = z:add(0.72,g)
	z = z:add(0.07,b)
	z = z:byte()
	y = torch.ByteTensor(1,z:size(1),z:size(2))
	y[1] = z
	return y
end

files = {}
i=0
for file in paths.files('eval_images',isImage) do
	if string.match(file,'.jpg') then
		i=i+1
		table.insert(files,path.join('eval_images',file))
	end
end
c = torch.Tensor(i,1,32,32)
i=0
for k,v in pairs(files) do
	i = i+1
	img = image.load(v)
	img = img:mul(255)
	img = img:byte()
	if img:size(1)==3 then
		img = rgb2gray(img)
	end
	img = image.scale(img,32,32)
	img = img:double()
	img = img:add(-mean)
	img = img:div(stdv)
	c[i] = img
end

sanskrit = {}
table.insert(sanskrit,'अ')
table.insert(sanskrit,'आ')
table.insert(sanskrit,'इ')
table.insert(sanskrit,'ई')
table.insert(sanskrit,'उ')
table.insert(sanskrit,'ऊ')
table.insert(sanskrit,'ऋ')
table.insert(sanskrit,'ए')
table.insert(sanskrit,'ऐ')
table.insert(sanskrit,'ओ')

prediction = {}
output = model:forward(c)
for i=1,output:size(1) do
	p,j  = torch.sort(output[i],true)
	entry = {image_id = files[i], caption = sanskrit[j[1]]}
	table.insert(prediction,entry)
	local cmd = 'cp "' .. files[i] .. '" vis/imgs/img' .. #prediction .. '.jpg' 
    print(cmd)
    os.execute(cmd) 
end

text = cjson.encode(prediction)
file = io.open('vis/vis.json','w')
file:write(text)
file:close()




