# semantic-segmentation-tensorflow
This is a Tensorflow implementation of semantic segmentation models on [MIT ADE20K scene parsing dataset](https://github.com/hangzhaomit/semantic-segmentation-pytorch) and [Cityscapes dataset](https://www.cityscapes-dataset.com/benchmarks/)
  
We re-produce the inference phase of several models, including [PSPNet](https://github.com/hszhao/PSPNet), [FCN](https://github.com/CSAILVision/sceneparsing), and [ICNet](https://github.com/hszhao/ICNet) by transforming the released pre-trained weights into tensorflow format, and apply on handcraft models. Also, we refer to ENet from [freg856 github](https://github.com/fregu856/segmentation). Still working on task integrated.

## Models
1. [PSPNet](https://github.com/hellochick/PSPNet-tensorflow)
2. FCN
3. ENet 
4. [ICNet](https://github.com/hellochick/ICNet-tensorflow)

### ...to be continue

## Install
Get corresponding transformed pre-trained weights, and put into `model` directory:   

 FCN       |PSPNet           |ICNet
|:-----------:|:-------------:|:------:|
|[Google drive](https://drive.google.com/file/d/1WElbk7ogK3e3-yEDP0yXfy4sCpbYL4yP/view?usp=sharing) | [Google drive](https://drive.google.com/file/d/17lcRDt-aJrr4fMom8cWJjAPhoGd911FS/view?usp=sharing)| [Google drive](https://drive.google.com/file/d/1Vg8NFk_k6Me7WSdXnDcDoFa4Pd0hl8tn/view?usp=sharing)|

## Inference
Run following command:
```
python inference.py --img-path /Path/To/Image --dataset Model_Type
```
### Arg list
```
--model - choose from "icnet"/"pspnet"/"fcn"/"enet"  
```

### Import module in your code:
```python
from model import FCN8s, PSPNet50, ICNet, ENet

model = PSPNet50() # or another model

model.read_input(img_path)  # read image data from path

sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

model.load(model_path, sess)  # load pretrained model
preds = model.forward(sess) # Get prediction 
```

## Results
### ade20k
|Input Image| PSPNet | FCN |  
:----------:|:------:|:----:
|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/input/indoor_1.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/pspnet_indoor_1.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/fcn_indoor_1.jpg)|  
|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/input/indoor_2.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/pspnet_indoor_2.jpg)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/fcn_indoor_2.jpg)|

### cityscapes
|Input Image|ICNet| ENet |
:----------:|:------:|:-----:|
|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/input/outdoor_1.png)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/icnet_outdoor_1.png)|![](https://github.com/hellochick/semantic-segmentation-tensorflow/blob/master/output/enet_outdoor_1.png)|

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
    
