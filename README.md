# semantic-segmentation-tensorflow
This is a Tensorflow implementation of semantic segmentation models on [MIT ADE20K scene parsing dataset](https://github.com/hangzhaomit/semantic-segmentation-pytorch).   
  
We re-produce the inference phase of [PSPNet](https://github.com/hszhao/PSPNet) and [FCN](https://github.com/CSAILVision/sceneparsing) by transforming the released pre-trained weights into tensorflow format, and apply on handcraft models.

## Install
Get corresponding transformed pre-trained weights, and put into `model` directory:   

 PSPNet       |FCN           |
|:-----------:|:-------------:|
|[Google drive](https://drive.google.com/file/d/1WElbk7ogK3e3-yEDP0yXfy4sCpbYL4yP/view?usp=sharing) | [Google drive](https://drive.google.com/file/d/17lcRDt-aJrr4fMom8cWJjAPhoGd911FS/view?usp=sharing)|

## Inference
Run following command:
```
python inference.py --img-path /Path/To/Image --dataset pspnet or fcn
```

### Import module in your code:
```python
from model import FCN8s, PSPNet50
from tools import *

model = PSPNet50() # model = FCN8s()

model.read_input(args.img_path)  # read image data from path

sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

model.load(model_path[args.model], sess)  # load pretrained model
preds = model.forward(sess) # Get prediction 
```

## Results

|Input Image| PSPNet | FCN |  
:----------:|:------:|:----:
|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/input/indoor_1.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/pspnet_indoor_1.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/fcn_indoor_1.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/input/indoor_2.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/pspnet_indoor_2.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/fcn_indoor_2.jpg)|



