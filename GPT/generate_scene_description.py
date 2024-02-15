from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

'''you need to save your OpenAI API key in the '.env' file, like:
    OPENAI_API_KEY=abc123 '''
# Load environment variables from .env file
load_dotenv()

client = OpenAI()

try:
    with open("text.txt", "r") as file:
        input_text = file.read()

    # Check input text length against OpenAI API limits
    if len(input_text.split()) > 4096:
        raise ValueError("Input text exceeds maximum token limit for GPT-3.5-turbo model.")

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant, you understand stories."},
            {"role": "user", "content": input_text}
        ]
    )

    content = completion.choices[0].message.content
    motions = content.split(';')
    
    # Process the motions: remove empty spaces, remove empty string
    motions = [motion.strip().replace("\n", "") for motion in motions \
                        if motion.strip().replace("\n", "") != ""]
    
    if len(motions) == 0:
        raise ValueError("Extract 0 number of human motion.")

    print(motions)


except OpenAIError as e:
    print("OpenAI API error:", e)

except ValueError as e:
    print("Error:", e)

except Exception as e:
    print("An unexpected error occurred:", e)
