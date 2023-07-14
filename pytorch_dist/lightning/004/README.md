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

