# Sign Language translation
inspired by [Nick Nochnack](https://github.com/nicknochnack)

# Setup
1. Install UV - `pip install uv`
2. Install all the dependencies `uv sync`


**Note:** The pretrained model is automatically downloaded from Hugging Face Hub on first run and cached locally.

# Running
To run in real time, run `uv run src/realtime.py`</br>

<strong>N.B.</strong> you might need need to update your camera parameter in cv2.VideoCapture() to get the right webcam for your machine.

# Using Your Own Pretrained Model

To train and share your own model with the team:

1. **Upload your model to Hugging Face Hub:**
```bash
uv run scripts/push_to_hf.py --repo-id YOUR_USERNAME/signdetr-pretrained
```

2. **Update the model path in `src/train.py` (line 29):**
```python
model.load_pretrained_from_hf('YOUR_USERNAME/signdetr-pretrained', '4426_model.pt', device=device)
```

3. **Commit and push the change** so your team uses your model.

The model will be automatically downloaded and cached on first run. No need to commit large model files to Git! 

# Collecting images 
1. Update classes in `src/utils/collect_images.py`
2. Run the script `uv run src/utils/collect_images.py`

# Labelling them 
1. Make sure label-studio is installed `uv pip list | grep label-studio`
2. Run the labelling tool `uv run label-studio`
3. Create new project, setup 
4. Labelling shortcuts CTRL + Enter submit, enter number per label 

# Training
1. Create a checkpoints folder `mkdir checkpoints`
2. Run the training pipeline `uv run src/train.py`
