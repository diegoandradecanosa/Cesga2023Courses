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

# Actividad NVBoard

Conéctate al FT3 y a un nodo con una GPU

```bash
ssh tuusuario@ft3.cesga.es
compute --gpu
```

Carga el entorno *mytf*

```bash
source $STORE/mytf/bin/activate
```

Inicia jupyter lab

```bash
jupyter lab --now-browser --ip=`hostname -i`
```

**En Jupyter:**

Arranca un terminal del entorno mytf en jupyterlab (File->New->Terminal)

```bash
cd $STORE/Cesga2023Courses/tf_dist/TF/001
source $STORE/mytf/bin/activate
python OneNodeResnet50TrainingTF.py 
```

Mientras se ejecuta el script anterior, observa el uso de la GPU a través de las funciones de NVDashBoard (Controles en la derecha de jupyterlab)

# Actividad una GPU

Conéctate al FT3 y a un nodo con una GPU, luego carga el entorno mytf

```bash
ssh tuusuario@ft3.cesga.es
compute --gpu
source $STORE/mytf/bin/activate
```

Ejecuta el script

```bash
cd $STORE/Cesga2023Courses/tf_dist/TF/001
python OneNodeResnet50Training.py
```





