# LLM-TSFD partial code
I am currently uploading code for data processing and visualization. The data used in this thesis consists of enterprise-simulated mock data.

## src
The LLM model directory includes OpenAI models as well as locally stored models.

### prompt
Some prompts are located in the __init__.py file within the src directory. The overall pipeline is outlined in the process flow within the __init__.py file.

### LLM(local LLM; openai models)
The model interfaces are located in the llm file, with all models adhering to the OpenAI model interface format. It also supports integration with locally deployed large models.

## mock_data
The dataset includes enterprise mock data, containing S1-S4 upper roller current data and crystallizer data.
