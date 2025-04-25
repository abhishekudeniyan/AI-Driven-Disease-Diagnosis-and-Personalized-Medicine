
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the LLM (Gemini model)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="****************************")  # Replace with your valid key

# Define the prompt template for a more structured output
prompt_template = PromptTemplate.from_template(
    """Provide a clear and professional response in the following format for the disease {disease}:
    
    Disease Description: [Provide a brief overview of the disease]
    Allopathic Medication: [List standard medications or treatments for the disease]
    Precautions: [Suggest any necessary precautions or lifestyle changes]
    
    Format each section as shown and be concise and professional."""
)

# Create a chain that connects the prompt with the LLM
chain = LLMChain(prompt=prompt_template, llm=llm)

# Function to fetch disease information
def getAnswer(disease: str) -> str:
    """Fetch structured information on a disease."""
    try:
        # Generate response
        response = chain.run({"disease": disease})
        return response  # Return the response directly
    except Exception as e:
        return f"An error occurred: {e}"

# Test the function independently
if __name__ == '__main__':
    disease = "SkIN RaSh"
    print(getAnswer(disease))

 