# Actividad Tensorboard 

Conéctate al FT3 y a un nodo con una GPU

```bash
ssh tuusuario@ft3.cesga.es
compute --gpu
```

Carga el entorno *mytf*

```bash
source $STORE/mytf/bin/activate
```

Inicia jupyterlab

```bash
export LD_LIBRARY_PATH=$STORE/mytf/lib/python3.9/site-packages/nvidia/cuda_cupti/lib/:$LD_LIBRARY_PATH
jupyter lab --no-browser --ip=`hostname -i`
```

Sigue el notebook Cesga2023Courses/tf_dist/TF/001/TensorBoard.ipynb

Inicia TensorBoard en el terminal

```bash
export LD_LIBRARY_PATH=$STORE/mytf/lib/python3.9/site-packages/nvidia/cuda_cupti/lib/:$LD_LIBRARY_PATH
jupyter lab --no-browser --ip=`hostname -i`
```

Visualiza la sesión de profiling que acabas de generar



