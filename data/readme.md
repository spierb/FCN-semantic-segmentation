<!--
 * @Author: bg
 * @Date: 2020-11-11 16:26:56
 * @LastEditTime: 2020-11-11 19:12:03
 * @LastEditors: bg
 * @Description: 
 * @FilePath: /FCN-semantic-segmentation/data/readme.md
-->
# data
## dataset index
- Cityscapes
- Pascal_voc
***
### Cityscapes 
```
.
├── gtFine_trainvaltest
│   ├── test
│   │   ├── berlin
│   │   └── ...
│   ├── train
│   │   ├── aachen
│   │   └── ...
│   └── val
│       ├── frankfurt
│       └── ...
├── leftImg8bit_trainvaltest
│   ├── test
│   ├── train
│   └── val
└── scripts
```
原图存放在leftImg8bit文件夹中，精细标注的数据存放在gtFine文件夹中.
- _gtFine_polygons.json存储的标注的第一手数据，即类（"label": "sky","building","sidewalk",等）及其在图像中对应的区域（由多边形"polygon"顶点在图像中的像素坐标给出的封闭区域）
- _gtFine_labelIds.png的值是0-33，不同的值代表不同的类，值和类的对应关系在Cityscapesscripts/helpers/labels.py中定义
- _gtFine_instaceIds.png是实例分割的结果，即对同一类中的不同个体进行区分
- _gtFine_color.png是为了可视化，不同类别与色彩的对应关系也在labels.py文件中给出。

https://blog.csdn.net/Cxiazaiyu/article/details/81866173
https://blog.csdn.net/qq_39350172/article/details/109548539
