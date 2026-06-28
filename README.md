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

### Step 4
needs:

# pip3 install openai
# pip install open
# node --version
# npm install openai

step_4_llm.py
put Prompts in .txt into prompt folder.
Terminal: 

bash.
export OPENAI_API_KEY="##key_link###"

bash: 
echo $OPENAI_API_KEY 

bash:
python -c "
from openai import OpenAI
client = OpenAI()
r = client.responses.create(
    model='gpt-4o-mini',
    input='Sag nur: Test erfolgreich'
)
print(r.output_text)
"

##select parameters and model 
bash:
python step_4_llm.py \
  --provider openai \
  --model gpt-4o-mini \
  --batch-size 40 \
  --context-size 3 \
  --temperature 0

#if prompts got actualized, make sure to clean cache
bash: 
rm -rf llm_cache_step4
bash: 
rm -rf step_4_output
