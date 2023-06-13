#!/bin/bash
#. /etc/profile.d/modules.sh


module load cuda11.7/toolkit/11.7.0-1
module load cudnn/8.2.0.53_cuda11.x
module load gcc/9.3.0
module load ffmpeg


SRC_PATH=/home/${USER}/multiVENT/multiCLIP/

FRAMES=12
CONFIG_PATH=/home/${USER}/multiVENT/multiCLIP/scripts/openclip_xlm
CONFIG_NAME=openclip

CHECKPOINT_PATH=/exp/${USER}/video/laion_models/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/open_clip_pytorch_model.bin
TOKENIZER_PATH=/exp/${USER}/video/laion_models/XLMRobertaTokenizerFast

DATA_PATH=/home/${USER}/data
DATA_FILE=multivent_ver_1/

NBR_GPUS=1
EXP_DIR=/exp/${USER}/video/openclip_xlm

FEAT_FILE="TRIAL_3_VIDEO_ALL"

mkdir -p ${EXP_DIR}
cd ${EXP_DIR}
FEAT_FILE=/exp/${USER}/video/feats

echo "NBR_GPUS - ${NBR_GPUS}"
echo "EXP_DIR - ${EXP_DIR}"
echo "USER - ${USER}"
echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "TMPDIR - ${TMPDIR}"
echo "SRC_PATH - ${SCRIPTS_PATH}"
echo "CONFIG_PATH - ${CONFIG_PATH}"
echo "CONFIG_NAME - ${CONFIG_NAME}"



cp -r ${DATA_PATH}/${DATA_FILE} ${TMPDIR}
cd ${TMPDIR}
tar -xf ${TMPDIR}/${DATA_FILE} 


python ${SRC_PATH}/scripts/openclip_xlm/openclip_featpool_multivent_infer.py \
--config-path ${CONFIG_PATH} \
--config-name ${CONFIG_NAME} \
checkpoint_path=${CHECKPOINT_PATH} \
tokenizer_path=${TOKENIZER_PATH} \
multivent_metadata_path=${TMPDIR}/multivent_ver_1/multivent_video_updated.csv \
video_retrieval_path=${DATA_PATH}/label-studio-data \
multivent_category_path=${TMPDIR}/multivent_ver_1/categories.csv \
multivent_event_path=${TMPDIR}/multivent_ver_1/events.csv \
multivent_language_path=${TMPDIR}/multivent_ver_1/languages.csv \
max_frames=${FRAMES} \
output=${EXP_DIR} \
feature_file_name=${FEAT_FILE}



python ${SRC_PATH}/src/video_retrieval/cli/retrieval_score.py \
--config-path ${CONFIG_PATH} \
--config-name ${CONFIG_NAME} \
feature_file_name=${FEAT_FILE} \
output=${EXP_DIR}

