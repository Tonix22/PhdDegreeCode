#!/bin/bash
# Script para ejecutar MATLAB en modo sin display y cargar configuración desde JSON

# Definir la ruta del script de MATLAB
MATLAB_GEN_SCRIPT_PATH="../../MatlabCode/MIMO/DPSKStack.m"
MATLAB_TEST_SCRIPT_PATH="../../MatlabCode/MIMO/MimoDPSKStackTest.m"
# Relative path to matlab file
JSON_PATH="../../Rutines/MIMO/NoV2VGen.json"
#Python scripts path
PYTHON_TRAINNING_PATH="../../PythonCode/DeepLearning/MIMOSolution/"
PYTHON_SCRIPT_PATH="${PYTHON_TRAINNING_PATH}main.py"
PYTHON_DATA_BASE="${PYTHON_TRAINNING_PATH}/data"

PYTHON_VISUALIZE_DATA_PATH="../../PythonCode/App/BERplots.py"

LIGHTNING_LOGS="lightning_logs/"
TRAINNED_MODELS="TrainnedModels/"

# Comprobar si el comando conda está disponible
if command -v conda >/dev/null 2>&1; then
    echo "Activando el entorno base de Conda..."
    # Se carga el script de configuración de Conda (ajusta la ruta si es necesario)
    source "$(conda info --base)/etc/profile.d/conda.sh"
    # Activar el entorno base
    conda activate myenv
    echo "Entorno myenv activado."
else
    echo "Conda no se encuentra instalado o no está en el PATH."
fi


rm -rf $LIGHTNING_LOGS
rm -rf $TRAINNED_MODELS
mkdir $TRAINNED_MODELS

# Ejecutar MATLAB en modo sin interfaz gráfica
matlab -nodisplay -nosplash -r "setenv('jsonPath', '$JSON_PATH'); run('$MATLAB_GEN_SCRIPT_PATH'); exit;"

# Ejecuta Entramiento en python
python $PYTHON_SCRIPT_PATH configTrainning.json

#Preload pytorch shared matlab enviorment
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

#Run testing enviroment
matlab -nodisplay -nosplash -r "setenv('jsonPath', '$JSON_PATH'); run('$MATLAB_TEST_SCRIPT_PATH'); exit;"

python $PYTHON_VISUALIZE_DATA_PATH configVisualize.json