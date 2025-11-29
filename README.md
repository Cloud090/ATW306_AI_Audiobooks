# README — Emotional TTS Model Training and Inference

This branch contains final versions of all scripts used to prepare datasets, fine-tune the Orpheus-3B TTS model, and run local inference tests. These scripts were developed for the ATW306 Emotional Storyteller project and document the full workflow behind the model component.

The scripts here are not part of the final deployed pipeline used by application.
Instead, they serve as supporting material demonstrating the model development work, data handling, and testing.

## Contents
### 1. Dataset Preparation
##### `create_cremad_dataset.py`

- Parses the CREMA-D dataset and converts files into a clean CSV format.
- Extracts:
    - Speaker ID
    - Emotion
    - Intensity
    - Sentence text
    - Audio file path
- Used as the base dataset for emotional fine-tuning.


##### `create_steven_dataset.py`

- Converts synthetic recordings (the custom “clear” speaker) into the same dataset format as CREMA-D.
- Trained with the same 6 emotions.
- Used to improve clarity and stabilise CREMA-D’s rougher speakers.

### 2. SNAC Encoding
`prep_snac.py`

- Loads the combined dataset CSVs.
- Resolves all audio paths + resamples audio to 24 kHz.
- Encodes audio into SNAC acoustic tokens (7-codes-per-frame format).
- Builds a HuggingFace `DatasetDict` containing:
    - Text
    - Speaker
    - Emotion
    - SNAC code stream
- Output is saved to disk and used during model training.

### 3. Model Training
`train.py`

- Loads the Orpheus-3B pretrained checkpoint via Unsloth.
- Applies LoRA adapters to Q/K/V/O and MLP layers.
- Packs text + acoustic codes into Orpheus format:
> `[START_HUMAN] text [END_HUMAN] [START_AI] <SNAC codes> [END_AI]`
- Trains for emotional/multi-speaker TTS using the SNAC dataset.
- Saves:
    - A final LoRA adapter
    - A merged FP16 model ready for inference
This is the script responsible for producing the model used in our pipeline.

### 4. Local Inference & Audio Generation
`run.py`

This is the main local generation script used to evaluate the model during development.
Features:
- Loads the merged TTS model (local or from HF).
- Loads SNAC for decoding acoustic tokens.
- Reads prompts from `prompts.json`.
- Supports emotional + speaker tags such as:
> `<spk=2001> <happy> <int_md> This is an example line.`
- Generates WAV files for each line.
- Cleans and normalises output.
- Optionally creates a combined audiobook-style file.

Output is stored under:
`outputs/runXX/out_###.wav`
`outputs/combined/combined_XX.wav`

### 5. How to Run Inference
1. Place your prompts in `data/prompts.json`
Supported formats:

> `["<spk=2001> <sad> <int_md> Hello there."]`
or
`[{"text": "<spk=1061> <happy> <int_md> Welcome!"}]`

2. Make sure the model is available:
- Local `models/...` folder
or
- HuggingFace repo (fallback)

3. Run:
`python run.py`

4. Generated audio will appear in:
`outputs/runXX/`

### Notes
> - These scripts were used throughout the development process to test model quality and shape the training pipeline. They are not deployed in the final product.
> - All scripts in this folder were developed and tested inside a dedicated Python virtual environment (venv).
> - No datasets or training checkpoints are included in this repository. Only the final merged model is uploaded publicly to HuggingFace for use in the project.
> - This README is included to clearly document the contribution, workflow, and reproducibility of the TTS component.
