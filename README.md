# U-Net for Precipitation Nowcasting

**U-Net architecture inspired by "Machine Learning for Precipitation Nowcasting from Radar Images".**

Google blog article: https://ai.googleblog.com/2020/01/using-machine-learning-to-nowcast.html

Original Google paper, Agrawal et al. 2019: https://arxiv.org/abs/1912.12132

Google U-net Architecture:
> ![Google U-net Architecture](https://1.bp.blogspot.com/-Mz4K8FlBjbE/Xh0CKBF8wOI/AAAAAAAAFMs/7r3_QnAhN9A0Ervr8plf7qVORnmFkh-qgCLcBGAsYHQ/s1600/image4.png)
> 
> (A) The overall structure of our U-NET. Blue boxes correspond to basic CNN layers. Pink boxes correspond to down-sample layers. Green boxes correspond to up-sample layers. Solid lines indicate input connections between layers. Dashed lines indicate long skip connections transversing the encoding and decoding phases of the U-NET. Dotted lines indicate short skip connections for individual layers. (B) The operations within our basic layer. (C) The operations within our down-sample layers. (D) The operations within our up-sample layers.
