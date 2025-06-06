# 普通diffusion

去噪过程就是预测噪声是N(0,1)中的那一个噪声

训练

![](https://files.hoshinorubii.icu/blog/2024/06/08/ddpm-training.jpg)


![](https://files.hoshinorubii.icu/blog/2024/06/08/denoising-unet.jpg)

训练成本高昂：模型在高维像素空间中执行生成过程，UNet通常具有约800M个参数，训练该模型需要数百个GPU天来，很容易耗费过多的算力来建模难以察觉的细节；另外，训练过程需要大量的无监督数据，这对于普通人来说是不可能完成的。

# Latent Diffusion Models (LDMs)

![](https://i-blog.csdnimg.cn/blog_migrate/2f886cb3a6d4697ba2f356746277c07c.png)

![](https://i-blog.csdnimg.cn/blog_migrate/d99b7fbec7e7eba2a10fb4384d11dce6.png)

![](https://i-blog.csdnimg.cn/blog_migrate/42f1c7817f0d867d24683c2ea4dd9309.png)

