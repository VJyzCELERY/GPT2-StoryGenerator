# CREDITS
Dataset was taken from books on [Project Gutenberg](https://www.gutenberg.org/) using [Gutendex API](https://gutendex.com/)<br>
The GPT model code was written and modified from [saqib707 on GitHub](https://github.com/saqib1707/gpt2-from-scratch)<br>
---

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/VJyzCELERY/GPT2-StoryGenerator
cd GPT2-StoryGenerator
```

### 2. Install dependencies
I recommend using a python venv as for the python version, the code was developed on python 3.12.12
```bash
python -m venv venv
source venv/bin/activate  
venv\Scripts\activate    
pip install -r requirements.txt
```

### 3. Running the app
to launch the gradio app simply do
```bash
python app.py
```

---
### Notes
if you want to train the model yourself, you may run the `gutenberg_downloader.ipynb` be sure to have ipykernel installed if not then you may run
```bash
pip install ipykernel
```
You may adjust and experiment with gutenberg downloader to get the amount of gutenberg book you want or get which book you want.<br>
After the dataset is created from the downloader, you may run the `dataprep.ipynb` then you can finally run the `train.ipynb`. Lastly, if you want to run your own model be sure to modify the line in the `app.py` for the model path specifically. Or if you want to just run it for testing you may use `inference.ipynb` and adjust the model_path accordingly.