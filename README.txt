训练数据集路径为data/train
测试数据集路径为data/test

训练命令:python train.py [--check_point] （模型参数文件） ps:如果添加模型参数，则将该参数载入模型，否则从零开始训练
测试命令:python train.py --check_point （模型参数文件） --do_train 0


代码结构
├── Inceptionv4-mindspore
    ├── new_check                   //模型参数，在运行中生成
    ├── data
    │   ├── test		//test数据集
    │   ├── train		//train数据集
    ├── README.txt		//所有模型的说明
    ├── config.py		//配置文件
    ├── CrossEntropy.py	//损失函数
    ├── dataloader.py	//及转化为dataset放入到model.train()中
    ├── inception_A.py	//inceptionv4中的模块
    ├── inception_B.py	//inceptionv4中的模块
    ├── inception_C.py	//inceptionv4中的模块
    ├── lr_generator.py
    ├── make_dict.py	//构造词典，将商品名编号
    ├── network.py                       
    ├── reduction_A.py	//inceptionv4中的模块
    ├── reduction_B.py	//inceptionv4中的模块
    ├── reduction_C.py	//inceptionv4中的模块
    ├── train.py		//输入do_train不为1时训练模型，否则为评价模型



