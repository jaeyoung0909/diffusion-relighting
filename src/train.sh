export MODEL_DIR="stabilityai/stable-diffusion-2-1-base"
export OUTPUT_DIR="./runs/test"   # output directory of each run
export DATA_PATH="./data/lighting_patterns"

accelerate launch src/train.py \
--seed=0 \
--pretrained_model_name_or_path=$MODEL_DIR \
--output_dir=$OUTPUT_DIR \
--dataset_name=$DATA_PATH \
--resolution=512 \
--learning_rate=1e-5 \
--validation_image "./data/conditioning_image_1.png" "./data/conditioning_image_2.png" \
--validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--set_grads_to_none \
--use_8bit_adam \
--checkpoints_total_limit 2 \
--validation_steps 100 \
--report_to "tensorboard"