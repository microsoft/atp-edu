# 知识点：

- DGGAN： deep conv  Generative Adversarial Nets.
- 教程参考[pytorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- 论文https://arxiv.org/abs/1511.06434
- 数据使用港中大的名人头像[数据集](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)。这里只有一类数据，就是名人头像

由于数据较大,没有放在git上，可以从[onedrive下载](https://microsoftapc.sharepoint.com/:u:/t/msteams_89c6ed/EVBSCIIK4epOtB1uyv14YbABc-H7D-WKmLVhfthDjIUfpQ?e=Lc6rqi),如果不适用云环境上课前需把数据单独提供给学生。 数据应解压到当前目录下celeba文件夹或再代码中修改路径




## DCGAN.ipynb
    1. 使用pytorch,回顾一下pytorch的特点，介绍一下pytorch处理图像数据的方法
    2. 介绍GAN的总体结构,训练流程
    3. 介绍反卷积（转置卷积）
    4. 分析一下loss变化情况，判别器输出的变化情况
    5. 重点（但是可以不讲），GAN训练包含多个不同文件，可以发现使用default Adam参数（beta 0.9 0.999）时效果差（这里效果差是指训练速度慢，可以看到defaultAdam20Batch效果也还可以的），可以何学员详细介绍下Adam，说明原因(GAN训练难，对超参数选择敏感) 可以参考https://jonathan-hui.medium.com/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b#:~:text=Listen-,GAN%20%E2%80%94%20Why%20it%20is%20so%20hard%20to%20train%20Generative%20Adversarial,Training%20GAN%20is%20also%20hard.
    6. 其实SGD也能完成任务（DCGAN-SGD）,只是训练时间要更长一些。
    7. RMSProp效果也还不错，训练速度也很快
    7. 这里多对于GAN评价都是主观的！！，没有客观指标