# Text Segmentation with TextTiling

This project performs text segmentation using the TextTiling algorithm from NLTK.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1 
step_1_texttiling.py needs: 
Goldstandard Text in "step_1_input" with the name "g_Standard.txt"

**Change Language**:
needs to be changed depending on task. Can be found in step_1_texttiling.py at the start. 
"""
from texttiling_de import TextTilingTokenizer
#from texttiling_eng import TextTilingTokenizer
"""

### Step 2 
step_2_evaluation_batch_distance.py needs: 
step_1_output --> should work automatically 
Goldstandard Grenzen in "Grenzen" with the name "g_grenzen.txt" 

### Step 3 
step_3_visualization_distance.py
--> should work automatically
