# Ejemplo simple DTENSORS

Este es un ejemplo simple de uso de DTENSORS, para probarlo debemos realizar los siguientes pasos.

1. Conectarse a FT3
2. Entrar en un nodo interactivo con 2 GPUs
3. Configurar el entorno mytf
4. Ejecuta el script dataparallel.py

```bash
ssh tusuario@ft3.cesga.es
source $STORE/Cesga2023Courses/pytorch_dist/scripts/interactive_1node_2gpus.sh
source $STORE/mytf/bin/activate
python dataparallel.py
```

El código ejecutado está basado en el [siguiente ejemplo](https://www.tensorflow.org/tutorials/distribute/dtensor_ml_tutorial).
Implementa una aplicación de *Sentiment Analysis* utilizando *DTENSORS*. Para ello:
- Crea un *Dataset* a través de datos compuestos por texto tokenizado.
- Construye un modelo *MLP* con capas especiales de tipo *Dense* y *BatchNorm* definidas a través de *tf.Module* que se usan para registrar las variables de inferencia.

Inspeccionamos ahora partes clave del código relacionadas con el entrenamiento distribuido. Por ejemplo, la definición de una capa *Dense*
en base a *DTensors*


https://github.com/diegoandradecanosa/Cesga2023Courses/blob/8974e0d39b4acd68cca64bddfb82b6bf70c4dfde/tf_dist/TF/006DTENSORS/dataparallel.py#L35-L63

En concreto, es importante inspeccionadr la parte donde se definen los pesos en base a *DTensors*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/8974e0d39b4acd68cca64bddfb82b6bf70c4dfde/tf_dist/TF/006DTENSORS/dataparallel.py#L43-L48

sucede lo mismo con los bias

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/8974e0d39b4acd68cca64bddfb82b6bf70c4dfde/tf_dist/TF/006DTENSORS/dataparallel.py#L56-L57

También es importante inspeccionar la definición de la capa *BatchNorm*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/8974e0d39b4acd68cca64bddfb82b6bf70c4dfde/tf_dist/TF/006DTENSORS/dataparallel.py#L66-L76

La definición de la arquitectura del modelo que las combina *MLP*

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/8974e0d39b4acd68cca64bddfb82b6bf70c4dfde/tf_dist/TF/006DTENSORS/dataparallel.py#L82-L97


Más adelante, creamos una Mesh con los dispositivos disponibles (en este caso 2 GPUs)

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/8974e0d39b4acd68cca64bddfb82b6bf70c4dfde/tf_dist/TF/006DTENSORS/dataparallel.py#L99

Luego, creamos una instancia del modelo distribuyendo los dtensors a través de la mesh

https://github.com/diegoandradecanosa/Cesga2023Courses/blob/8974e0d39b4acd68cca64bddfb82b6bf70c4dfde/tf_dist/TF/006DTENSORS/dataparallel.py#L101-L102





