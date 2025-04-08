#!/bin/bash
# Script para ejecutar MATLAB en modo sin display y cargar configuración desde JSON

# Definir la ruta del script de MATLAB
MATLAB_GEN_SCRIPT_PATH="../../MatlabCode/MIMO/DPSKStack.m"
MATLAB_TEST_SCRIPT_PATH="../../MatlabCode/MIMO/MimoDPSKStackTest.m"
# Relative path to matlab file
CONFIG_JSON_PATH="../../Rutines/MIMO/NoV2VGen.json"
# Python scripts path
PYTHON_TRAINNING_PATH="../../PythonCode/DeepLearning/MIMOSolution/"
PYTHON_SCRIPT_PATH="${PYTHON_TRAINNING_PATH}main.py"
PYTHON_DATA_BASE="${PYTHON_TRAINNING_PATH}/data"

PYTHON_VISUALIZE_DATA_PATH="../../PythonCode/App/BERplots.py"

LIGHTNING_LOGS="lightning_logs/"
TRAINNED_MODELS="TrainnedModels/"

# Parsear argumentos
GENERATE=false
TRAIN=false
TEST=false

while getopts "gtr" opt; do
    case $opt in
        g) GENERATE=true ;;
        t) TRAIN=true ;;
        r) TEST=true ;;
        *) echo "Uso: $0 [-g] [-t] [-r]" >&2; exit 1 ;;
    esac
done


# Ejecutar generación de datos si se especifica -g
if $GENERATE; then
    echo "Generando datos..."
    matlab -nodisplay -nosplash -r "setenv('jsonPath', '$CONFIG_JSON_PATH'); run('$MATLAB_GEN_SCRIPT_PATH'); exit;"
fi

# Ejecutar entrenamiento si se especifica -t
if $TRAIN; then
    # Crear carpetas necesarias
    rm -rf $LIGHTNING_LOGS
    rm -rf $TRAINNED_MODELS
    mkdir $TRAINNED_MODELS
    echo "Entrenando modelo..."
    python --version
    python $PYTHON_SCRIPT_PATH configTrainning.json
fi

# Ejecutar pruebas si se especifica -r
if $TEST; then
    #Preload pytorch shared matlab enviorment
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    echo "Ejecutando pruebas..."
    matlab -nodisplay -nosplash -r "setenv('jsonPath', '$CONFIG_JSON_PATH'); run('$MATLAB_TEST_SCRIPT_PATH'); exit;"
    python $PYTHON_VISUALIZE_DATA_PATH configVisualize.json
fi