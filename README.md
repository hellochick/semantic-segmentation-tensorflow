# semantic-segmentation-tensorflow
This is a Tensorflow implementation of semantic segmentation models on [MIT ADE20K scene parsing dataset](https://github.com/hangzhaomit/semantic-segmentation-pytorch).   
  
We re-produce the inference phase of [PSPNet](https://github.com/hszhao/PSPNet) and [FCN](https://github.com/CSAILVision/sceneparsing) by transforming the released pre-trained weights into tensorflow format, and apply on handcraft models.

## Models
1. PSPNet
2. FCN
3. ENet
...to be continue

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

model = PSPNet50() # model = FCN8s()

model.read_input(args.img_path)  # read image data from path

sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

model.load(model_path[args.model], sess)  # load pretrained model
preds = model.forward(sess) # Get prediction 
```

## Results
### ade20k
|Input Image| PSPNet | FCN |  
:----------:|:------:|:----:
|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/input/indoor_1.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/pspnet_indoor_1.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/fcn_indoor_1.jpg)|  
|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/input/indoor_2.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/pspnet_indoor_2.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/fcn_indoor_2.jpg)|

### cityscapes
|Input Image| ENet |
:----------:|:------:|
|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/input/outdoor_1.png)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/enet_outdoor_1.png)|

## Citation
    @inproceedings{zhao2017pspnet,
      author = {Hengshuang Zhao and
                Jianping Shi and
                Xiaojuan Qi and
                Xiaogang Wang and
                Jiaya Jia},
      title = {Pyramid Scene Parsing Network},
      booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
    }
Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }
    
Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. arXiv:1608.05442. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2016semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={arXiv preprint arXiv:1608.05442},
      year={2016}
    }
    
