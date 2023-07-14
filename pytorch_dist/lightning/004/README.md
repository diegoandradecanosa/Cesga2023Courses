# Ejemplo de conjuntos de datos propios

Este ejemplo ilustra el uso de *DataModules* en Lightning para crear nuestros conjuntos de datos personalizados.
El código está distribuido en dos ficheros:
- [mnist_datamodule.py](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/lightning/004/mnist_datamodule.py) Contiene la definición del 
*DataModule*. En este caso se trata de un *DataModule* para el *DataSet* MNIST.
- [mnist_sample.py](https://github.com/diegoandradecanosa/Cesga2023Courses/blob/main/pytorch_dist/lightning/004/mnist_sample.py) Contiene el código de un entrenamiento 
distribuido en Lightning que hace uso de ese *DataModule*.

Para ejecutar el ejemplo debemos seguir los siguientes pasos:
```
compute  --gpu
source $STORE/mytorchdist/bin/activate
python mnist_sample.py
```
- Pedimos un nodo con una GPU 
- Activamos el entorno del curso
- Ejecutamos el script directamente con Python

La definición de un *DataModule* propio en Lightning se realiza definiendo una clase que hereda de *LightningDataModule*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/c2897f5e2612cc606c1cd2d955b49f57d56074e2/pytorch_dist/lightning/004/mnist_datamodule.py#L147

Se deben definir implementaciones para todos o varios de estos elementos: *prepare_data,setup,train_dataloader,val_dataloader,test_dataloader* y *predict_dataloader*.

Veamos un ejemplo de implementación de uno de estos métodos para nuestro ejemplo

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/c2897f5e2612cc606c1cd2d955b49f57d56074e2/pytorch_dist/lightning/004/mnist_datamodule.py#L207-L216

El *DataModule* definido se usa en el código principal como un argumento más para llamar al *MNISTDataModule* 

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/c2897f5e2612cc606c1cd2d955b49f57d56074e2/pytorch_dist/lightning/004/mnist_sample.py#L56-L61

