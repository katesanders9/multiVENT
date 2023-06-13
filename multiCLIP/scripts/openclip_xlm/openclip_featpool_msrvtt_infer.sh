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

DATA_PATH=/home/${USER}/data/msrvtt
DATA_ZIP=MSRVTT.zip


NBR_GPUS=1
EXP_DIR=/exp/${USER}/video/openclip_xlm
FEAT_FILE=/exp/${USER}/video/feats

mkdir -p ${EXP_DIR}
cd ${EXP_DIR}

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


mkdir -p ${TMPDIR}/video
cd ${TMPDIR}/video

echo "copy ${DATA_PATH}/${DATA_ZIP} to ${TMPDIR}/video"
cp ${DATA_PATH}/${DATA_ZIP} ${TMPDIR}/video
unzip -q ${TMPDIR}/video/${DATA_ZIP} -d ${TMPDIR}/video


python ${SRC_PATH}/scripts/openclip_xlm/openclip_featpool_msrvtt_infer.py \
--config-path ${CONFIG_PATH} \
--config-name ${CONFIG_NAME} \
checkpoint_path=${CHECKPOINT_PATH} \
tokenizer_path=${TOKENIZER_PATH} \
test_csv=${TMPDIR}/video/MSRVTT/anns/MSRVTT_JSFUSION_test.csv \
data_path=${TMPDIR}/video/MSRVTT/anns/MSRVTT_data.json \
features_path=${TMPDIR}/video/MSRVTT/videos/all \
max_frames=${FRAMES} \
output=${TMPDIR} \
feature_file_name=${FEAT_FILE}


python ${SRC_PATH}/src/video_retrieval/cli/retrieval_score.py \
--config-path ${CONFIG_PATH} \
--config-name ${CONFIG_NAME} \
feature_file_name=${FEAT_FILE} \
output=${TMPDIR}
