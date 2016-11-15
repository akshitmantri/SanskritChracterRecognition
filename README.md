# SanskritChracterRecognition
A Lenet model along with the train data, test data and the pretrained model.

All the models and the script run on torch 7. The libraries used are image, nn, xlua and paths.
The data_prepro.lua file is optional as it converts .mat file to .t7. If you already have data present in .t7 format then you can directly use the lenet_model.lua for training.

The file lenet_model.lua first makes a lenet model which is commented in the code. To build and train the model from scratch just uncomment the model:add() lines and comment the torch.load() command.
The model is set to run for 5 epochs.

The script eval_lenet.lua takes .jpg images from the folder eval_images and predicts the respective sanskrit character. After running this script, to see the visual demo of the model type the following commands on the commandline.

```lua
cd vis
python -m SimpleHTTPServer
```
Open the browser and go to the link localhost:8000. the images along with the respective character is displayed.
