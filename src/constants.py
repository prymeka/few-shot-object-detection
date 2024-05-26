import os


SEED_START = 0
SEED_STOP = 20
SHOTS = (1, 2, 3, 5, 10, 30)

DATA_DIR = os.path.expanduser('~/.datasets/')

# COCO
COCO_DATA_DIR = DATA_DIR + 'coco/'
COCO_ANNOTATIONS_DIR = COCO_DATA_DIR + 'annotations/'

COCO_TRAIN_IMAGES = COCO_DATA_DIR+'train2017/'
COCO_TRAIN_ANNOTATIONS = COCO_ANNOTATIONS_DIR+'prepared_instances_train2017.json'
COCO_RAW_TRAIN_ANNOTATIONS = COCO_ANNOTATIONS_DIR+'instances_train2017.json'

COCO_VAL_IMAGES = COCO_DATA_DIR+'val2017/'
COCO_VAL_ANNOTATIONS = COCO_ANNOTATIONS_DIR+'prepared_instances_val2017.json'
COCO_RAW_VAL_ANNOTATIONS = COCO_ANNOTATIONS_DIR+'instances_val2017.json'

COCO_NUM_CATEGORIES = 90

# CPPE-5
CPPE_DATA_DIR = DATA_DIR + 'cppe5/'
CPPE_ANNOTATIONS_DIR = CPPE_DATA_DIR + 'annotations/'

CPPE_RAW_TRAIN = CPPE_DATA_DIR + 'train-00000-of-00001.parquet'
CPPE_TRAIN_IMAGES = CPPE_DATA_DIR + 'train/'
CPPE_TRAIN_ANNOTATIONS = CPPE_ANNOTATIONS_DIR + 'cppe_train.json'

CPPE_RAW_VAL = CPPE_DATA_DIR + 'test-00000-of-00001.parquet'
CPPE_VAL_IMAGES = CPPE_DATA_DIR + 'val/'
CPPE_VAL_ANNOTATIONS = CPPE_ANNOTATIONS_DIR + 'cppe_val.json'

CPPE_CATEGORIES = [
    {'supercategory': 'ppe', 'id': 1, 'name': 'coveralls'},
    {'supercategory': 'ppe', 'id': 2, 'name': 'face shield'},
    {'supercategory': 'ppe', 'id': 3, 'name': 'gloves'},
    {'supercategory': 'ppe', 'id': 4, 'name': 'goggles'},
    {'supercategory': 'ppe', 'id': 5, 'name': 'mask'},
]
CPPE_NUM_CATEGORIES = 5
CPPE_WEIGHTS_DIR = CPPE_DATA_DIR + 'weights/'
CPPE_PLOTS_DIR = CPPE_DATA_DIR + 'plots/'

CPPE_ANN_ID_ADDEND = 2232195
CPPE_IMG_ID_ADDEND = 581929

# FSL 
FSL_DATA_DIR = DATA_DIR + 'fsl/'
FSL_ANNOTATIONS_DIR = FSL_DATA_DIR + 'annotations/'
FSL_VAL_ANNOTATIONS = FSL_ANNOTATIONS_DIR+'fsl_val_all.json'
FSL_VAL_ANNOTATIONS_COCO = FSL_ANNOTATIONS_DIR+'fsl_val_coco.json'
FSL_VAL_ANNOTATIONS_CPPE = FSL_ANNOTATIONS_DIR+'fsl_val_cppe.json'

# Novel Training 
NOVEL_BATCH_SIZE = 16
NOVEL_LEARNING_RATE = 0.01

# Training
BATCH_SIZE = 16
LEARNING_RATE = 0.002
MOMENTUM = 0.9
DECAY = 0.0001

# Cosine Similarity Alpha
COSINE_SCALE = 20.0

# Error Analysis
ERROR_DIR = FSL_DATA_DIR + 'analysing_errors/'

