## Dataset Preparation

### VoiceBank + DEMAND Dataset

1. **Download** the VoiceBank + DEMAND dataset from  
   https://datashare.ed.ac.uk/handle/10283/2791  
   We used the **28 speakers** subset and the corresponding **test set**.

2. **Extract** the downloaded files.

3. **Generate the dataset JSON files** by running:

   `python data/make_dataset_json.py`

   Make sure the script paths match the location of your extracted files.  
   You can also use the provided JSON files if they match your setup.

---

### RIR Dataset (for Dereverberation)

4. **Generate synthetic Room Impulse Responses (RIRs)** by running:

   `python data/create_rirs_dataset.py`

   This step is required for training on the **dereverberation task**.
