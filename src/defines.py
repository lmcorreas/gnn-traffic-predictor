
DISTANCE_THRESHOLD=5
RANDOM_SEED=13


MAX_DAY_VALUE = 31.0
MAX_MONTH_VALUE = 12.0
MAX_YEAR_VALUE = 2022.0
MAX_HOUR_VALUE=23.0
MAX_MINUTE_VALUE = 59.0
MAX_DAY_MINUTES_VALUE=1440.0


#  0 - Lunes
#  1 - Martes
#  2 - Miércoles
#  3 - Jueves
#  4 - Viernes
#  5 - Sábado
#  6 - Dia
#  7 - Mes
#  8 - Año
#  9 - Hora
# 10 - Minuto
# 11 - Festivo
# 12 - Minutos desde 0:00
# 13 - Minutos hasta 0:00
# 14 - Previo a festivo
# 15 - Siguiente a festivo
# 16 - Enero
# 17 - Febrero
# 18 - Marzo
# 19 - Abril
# 20 - Mayo
# 21 - Junio
# 22 - Julio
# 23 - Agosto
# 24 - Septiembre
# 25 - Octubre
# 26 - Noviembre
# 27 - Diciembre

NUM_INPUT_FEATURES = 28


NUM_OUTPUT_VALUES = 3

# Número de valores de salida a ser considerados en función de pérdida
#NUM_CHECKED_VALUES = 3
NUM_CHECKED_VALUES = 1

# Salidas
# 0 - intensidad
MAX_INTENSITY_VALUE = 15000.0
# 1 - ocupacion
MAX_OCCUPANCY_VALUE = 100.0
# 2 - carga
#MAX_CHARGE_VALUE = 100.0
MAX_CHARGE_VALUE = 1.0
# 3 - vmed
# NOT APPLICABLE

NUM_HIDDEN_FEATURES_LINEAR = 50
NUM_HIDDEN_FEATURES_GNN = 150


NUM_GNN_LAYERS = 5
MODEL_PARAMS_PATH = './model/model.pt'
MODEL_VOIDING_PARAMS_PATH = './model/modelVoiding.pt'
MODEL_SAN_JOSE = './model/modelSanJose.pt'


VOID_SOME_POINTS_TRAINING = False


BACKEND = 'PYTORCH'
ACTIVATE_GPU = True


BATCH_SIZE = 16

SHUFFLE_TRAIN_DATA = False


LEARNING_RATE = 0.00005
LEARNING_RATE_GNN = 0.00005
DECAY_STEP = 1
GAMMA = 0.5

LOAD_MODEL = True
RECALC_MODEL = False

APPLY_LABELS_BEFORE_GNN = False

ONLY_COST_ON_GNN = True

NUM_EPOCHS = 1000

OUT_LIMITS_FACTOR = 1

USE_DOUBLE = False


# MAP
MAP_FROM_PLACE=False
MAP_PLACE = 'Madrid, Spain'
MAP_TYPE = 'drive'


LINEAR_MULT = 1
GNN_MULT = 1
INTER_MULT = 0





TRAINING_SIZE = 0.9
VAL_SIZE = 0.05

EPOCHS_WITHOUT_GNN = 0


EXPORT_PROCESSED_FILE = False

#GNN_INSTANCES_PER_EPOCH = 1024
GNN_INSTANCES_PER_EPOCH = 256

NUM_GNN_TO_TRAIN = 1

RANDOMIZE_NUM_GNN = False


SECOND_MAP = True

if SECOND_MAP:
    DATA_BASE_PATH= "data_pems_bay/"
    SENSOR_INFO_FILE_NAME = 'graph_sensor_locations_bay.csv'
    OUTPUT_BASE_PATH="output_pems_bay/"
    MIN_ID = 0
    MAX_ID = 0
    DISTANCE_THRESHOLD = 20
    LEARNING_RATE = 0.00001
    LEARNING_RATE_GNN = 0.000005
    LOAD_MODEL = True
    RECALC_MODEL = False
else:
    DATA_BASE_PATH= "data/"
    SENSOR_INFO_FILE_NAME = 'pmed_ubicacion_03-2022.csv'
    OUTPUT_BASE_PATH="output/"
    MIN_ID = 101    # OCT-2021
    MAX_ID = 106    # MAR-2022