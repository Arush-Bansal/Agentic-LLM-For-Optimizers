import numpy as np
from openai import OpenAI
import json
import re

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
client = OpenAI(
    api_key =  r'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'
)



def prepare_chain_of_thought_prompt(data_points, previous_guesses, previous_errors):
    """
    Prepare the prompt for the LLM's chain-of-thought reasoning.
    """
    prompt = "Given the following data points:\n\n"
    prompt += "x\ty\n"
    for x_i, y_i in data_points:
        prompt += f"{x_i:.3f}\t{y_i:.3f}\n"
    prompt += "\nYour task is to find the best fit values for A and B in the function y = A x^2 + B, to model the data.\n"
    prompt += "Please explain your step-by-step reasoning in determining the optimal values for A and B.\n"
    
    if previous_guesses:
        prompt += "\nPrevious guesses and their errors:\n"
        for i, (A_i, B_i) in enumerate(previous_guesses):
            error_i = previous_errors[i]
            prompt += f"Iteration {i+1}: A = {A_i:.6f}, B = {B_i:.6f}, Error = {error_i:.6f}\n"
        prompt += "\nBased on the previous attempts, analyze how to adjust A and B to minimize the error.\n"

    prompt += "Make the best use of 1000 tokens here. make inferences on the data provided to you.\n"

    return prompt

def prepare_final_answer_prompt():
    """
    Prepare the prompt for the LLM to provide the final answer.
    """
    prompt = "Now, please provide your final estimates for A and B without any additional explanation.\n"
    prompt += "Output your answer in the following format:\nA = [value]\nB = [value]\n"
    return prompt

def extract_values(response):
    """
    Extract A and B values from the LLM's response using regular expressions.
    """
    A_pattern = r"A\s*=\s*([-\d.Ee]+)"
    B_pattern = r"B\s*=\s*([-\d.Ee]+)"
    A_match = re.search(A_pattern, response)
    B_match = re.search(B_pattern, response)

    if A_match and B_match:
        try:
            A = float(A_match.group(1))
            B = float(B_match.group(1))
            return A, B
        except ValueError:
            return None, None
    else:
        return None, None

def compute_error(A, B, x, y):
    """
    Compute the sum of squared errors between the model and the data.
    """
    y_pred = A * x**2 + B
    error = np.sum((y - y_pred)**2)
    return error


for super_iteration in range(10):
    
    
    # Generate synthetic data
    np.random.seed(42)  # For reproducibility
    x = np.linspace(-5, 5, 10)
    true_A = 2
    true_B = 3
    noise = np.random.normal(0, 1, x.shape)
    y = true_A * x**2 + true_B + noise
    data_points = list(zip(x, y))
    
    
    # Initialize lists to store previous guesses and errors
    previous_guesses = []
    previous_errors = []
    data_record = []


    
    
    
    # Iterative loop
    for iteration in range(10):
        print(f"Iteration {iteration+1}:")

        # Prepare the chain-of-thought prompt for the LLM
        cot_prompt = prepare_chain_of_thought_prompt(data_points, previous_guesses, previous_errors)
        # print(cot_prompt)

        # Call the OpenAI API for chain-of-thought reasoning
        try:


            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": cot_prompt}
                ],
                max_tokens=1000,
                temperature=0.7,
                stream=True
            )


            cot_reply = "".join([str(chunk.choices[0].delta.content) for chunk in completion])
            print(cot_reply)
            
            
            
            # cot_response = OpenAI.ChatCompletion.create(
            #     model="gpt-4",
            #     messages=[
            #         {"role": "user", "content": cot_prompt}
            #     ],
            #     max_tokens=500,
            #     temperature=0.7,
            # )
        except Exception as e:
            print(f"An error occurred during chain-of-thought API call: {e}")
            break

    #     # Get the assistant's chain-of-thought reply
    #     # cot_reply = cot_response['choices'][0]['message']['content']
    #     print("Chain-of-thought reasoning:")
    #     print(cot_reply)

        # Prepare the final answer prompt
        final_prompt = prepare_final_answer_prompt()

        # Call the OpenAI API for the final answer
        try:
            
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": cot_prompt},
                    {"role": "assistant", "content": cot_reply},
                    {"role": "user", "content": final_prompt},
                ],
                max_tokens=200,
                temperature=0.0,
                stream=True
            )
            assistant_reply = "".join([str(chunk.choices[0].delta.content) for chunk in completion])
            print(assistant_reply)
            # final_response = openai.ChatCompletion.create(
            #     model="gpt-4",
            #     messages=[
            #         {"role": "user", "content": final_prompt}
            #     ],
            #     max_tokens=50,
            #     temperature=0.0,
            # )
        except Exception as e:
            print(f"An error occurred during final answer API call: {e}")
            break

    #     # Get the assistant's final reply
    #     assistant_reply = final_response['choices'][0]['message']['content']
    #     print("Assistant's final answer:")
    #     print(assistant_reply)

        # Extract A and B from the assistant's reply
        A_guess, B_guess = extract_values(assistant_reply)

        if A_guess is None or B_guess is None:
            print(f"Couldn't extract A and B in iteration {iteration+1}. Assistant's reply:")
            print(assistant_reply)
            break  # Exit the loop if extraction fails

        # Compute the error for the current guesses
        error = compute_error(A_guess, B_guess, x, y)

        # Store the guesses and error for future iterations
        previous_guesses.append((A_guess, B_guess))
        previous_errors.append(error)

        # Record data for this iteration
        record = {
            'iteration': iteration + 1,
            'A': A_guess,
            'B': B_guess,
            'error': error,
            'chain_of_thought': cot_reply,
            'assistant_reply': assistant_reply
        }
        data_record.append(record)

        # Print the results of the current iteration
        print(f"Estimated A = {A_guess:.6f}, B = {B_guess:.6f}, Error = {error:.6f}\n")

    # Save the data records to a JSON file
    with open(f'data_record_{super_iteration}_gpt4o.json', 'w') as f:
        json.dump(data_record, f, indent=4)