# LLAMAIndex

This is a sample project for learning how to use LLAMAindex and other prompting approaches using LLM. It is intended for educational purposes only and should not be used in production environments. The code imports the necessary libraries and defines a function that uses LLAMAindex to generate text prompts. The function takes a prompt as input and returns a generated text response. To use this code, simply run the function with a prompt as input and the generated text response will be returned.


#### Define function to generate text prompts using LLAMAindex
```

import openai
import json


def generate_text(prompt):
    # Set up OpenAI API credentials
    with open('secrets.json') as f:
        secrets = json.load(f)
    openai.api_key = secrets['api_key']

    # Set up LLAMAindex model and generate text prompt
    model_engine = "text-davinci-002"
    prompt = (f"{prompt}\n\nResponse:")
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Return generated text response
    return response.choices[0].text
```

