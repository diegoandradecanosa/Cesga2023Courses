# How to use Tensorflow with Ray Train

```bash
sbatch sbatch_ngpus.sh training.py
```

Este tutorial demuestra cómo realizar un entrenamiento distribuido de varios trabajadores con un modelo Keras y Ray.

## Definición de conjunto de datos y modelo.
A continuación, cree una configuración simple de modelo y conjunto de datos MNIST.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/f0fcf1011c3c62034f376200b5bbf47ab42a935d/tf_dist/TF/007/training.py#L17-L30

Utiliza tf.data to separar por lotes y mezclar el conjunto de datos:

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/f0fcf1011c3c62034f376200b5bbf47ab42a935d/tf_dist/TF/007/training.py#L25

Construye el modelo tf.keras utilizando la API de Keras model subclassing API:

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/f0fcf1011c3c62034f376200b5bbf47ab42a935d/tf_dist/TF/007/training.py#L33-L44

Escoge un optimizador y una funcion de perdida para el entrenamiento de tu modelo:

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/f0fcf1011c3c62034f376200b5bbf47ab42a935d/tf_dist/TF/007/training.py#L65-L66

Escoge metricas para medir la perdida y exactitud del modelo. Estas metricas acumulan los valores cada epoch y despues imprimen el resultado total.

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/f0fcf1011c3c62034f376200b5bbf47ab42a935d/tf_dist/TF/007/training.py#L67

Utiliza el método ```TensorflowTrainer``` para entrenar la configuración creada de forma distribuida con Ray

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/f0fcf1011c3c62034f376200b5bbf47ab42a935d/tf_dist/TF/007/training.py#L87-L93