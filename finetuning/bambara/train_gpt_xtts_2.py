import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
from bambara_training_utils import BambaraGPTTrainer, bambara_dataset_formatter, build_reference_audios_dict

# Logging parameters
RUN_NAME = "xtts_lr_1e-05_mel_loss_1.5_epochs_40"
PROJECT_NAME = "BAM_FINE_TUNING_3"
DASHBOARD_LOGGER = "wandb"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = False  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 12  # set here the batch size
GRAD_ACUMM_STEPS = 21  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

# Define here the dataset that you want to use for the fine-tuning on.
config_dataset = BaseDatasetConfig(
    formatter=bambara_dataset_formatter,
    dataset_name="bambara_tts",
    path="./dataset",
    meta_file_train="../dataset/metadata.txt",
    meta_file_val="../dataset/metadata_val.txt",
)

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [config_dataset]

# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE files
DVAE_CHECKPOINT_LINK = "https://huggingface.co/oza75/bambara-tts/resolve/main/dvae.pth"
MEL_NORM_LINK = "https://huggingface.co/oza75/bambara-tts/resolve/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://huggingface.co/oza75/bambara-tts/resolve/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://huggingface.co/oza75/bambara-tts/resolve/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
# TOKENIZER_FILE = "./saved/combined_vocab.json"  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

# download XTTS v2.0 files if needed
if not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

# Training sentences generations
# speaker reference to be used in training test sentences
SPEAKER_REFERENCES = build_reference_audios_dict("./reference_audios")


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        gpt_loss_text_ce_weight=0.01,
        gpt_loss_mel_ce_weight=1.5,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
        epochs=40,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=100,
        plot_step=100,
        log_model_step=1194,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        mixed_precision=True,
        use_grad_scaler=True,
        grad_clip=7.0,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2, "fused": True},
        lr=1e-05,  # learning rate
        # lr_scheduler="MultiStepLR",
        lr_scheduler="ExponentialLR",
        warmup_steps=1193 * 3,
        warmup_start_lr=0.1,
        # it was adjusted accordly for the new step scheme
        # lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        lr_scheduler_params={"gamma": 0.01, "last_epoch": -1},
        transliterate_bambara=False,
        sound_norm_refs=True,
        distributed_url="tcp://127.0.0.1:54321",
        test_sentences=[
            {
                "text": "Dumuni bɛ taa farikolo fan jumɛn ?",
                "speaker_wav": SPEAKER_REFERENCES['bm'][0],
                "language": 'bm',
            },
            {
                "text": "Ni sumaya furakɛli daminɛna, an ka kan ka to ka fura ta ka taa ɲɛ, walima ka to ka pikiri ni sɔrɔmuw kɛ ka taa ɲɛ fo sumaya ka ban pew.",
                "speaker_wav": SPEAKER_REFERENCES['bm'][1],
                "language": 'bm',
            },
            {
                "text": "A ko kɛra degunba ye jamanadenw ma kɛrɛnkɛrɛnna demisɛn finitiniw ni mɔgɔ kɔrɔbaw.",
                "speaker_wav": SPEAKER_REFERENCES['bm'][2],
                "language": 'bm',
            },
            {
                "text": "Silamɛ dannabaaw Burkina Faso la, u ye Eid El Fitr seli kɛ seli la min kɛra sun kalo laban don na .",
                "speaker_wav": SPEAKER_REFERENCES['bm'][3],
                "language": 'bm',
            },
            {
                "text": "le texte devra attendre l’avis du Conseil constitutionnel avant son examen à l’Assemblée.",
                "speaker_wav": SPEAKER_REFERENCES['fr'][0],
                "language": 'fr',
            },
            {
                "text": "Below are benchmarks for downsampling and upsampling waveforms between two pairs of sampling rates.",
                "speaker_wav": SPEAKER_REFERENCES['en'][0],
                "language": 'en',
            },
            {
                "text": "La convivencia se asienta en Euskadi con la asignatura pendiente de la memoria",
                "speaker_wav": SPEAKER_REFERENCES['es'][0],
                "language": 'es',
            },
            {
                "text": "Quei mariuoli di troppo alla corte dell’ex sceriffo. Così il sistema Emiliano sta affondando la Puglia",
                "speaker_wav": SPEAKER_REFERENCES['it'][0],
                "language": 'it',
            },
            {
                "text": "Les Insoumis ont obtenu ce mardi 9 avril que le texte soit retiré de l’ordre du jour de l’Assemblée nationale en attendant un avis du Conseil constitutionnel.",
                "speaker_wav": SPEAKER_REFERENCES['fr'][0],
                "language": 'fr',
            },
        ],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)
    # model = torch.compile(model, fullgraph=True)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        formatter=bambara_dataset_formatter,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
            use_accelerate=False,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
