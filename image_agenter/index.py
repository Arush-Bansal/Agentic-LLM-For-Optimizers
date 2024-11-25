import numpy as np
import matplotlib.pyplot as plt
from  openai import OpenAI
import json
import re

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
client = OpenAI(
    api_key =  r'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'
)

# Generate synthetic data
np.random.seed(42)  # For reproducibility
x = np.linspace(-5, 5, 10)
true_A = 2
true_B = 3
noise = np.random.normal(0, 1, x.shape)
y = true_A * x**2 + true_B + noise
data_points = list(zip(x, y))

def generate_plot_description(x, y):
    """
    Generate a textual description of the plot.
    """
    description = "The plot depicts a set of data points representing y versus x.\n"
    description += f"The x-values range from {x.min():.2f} to {x.max():.2f}.\n"
    description += "The data seems to follow a quadratic trend.\n"
    return description

def prepare_prompt(description, previous_guesses, previous_errors):
    """
    Prepare the prompt message to send to the LLM.
    """
    prompt = "Given the following description of a plot:\n\n"
    prompt += description + "\n"
    prompt += "Your task is to find the best fit values for A and B in the function y = A x^2 + B to model the data.\n"

    if previous_guesses:
        prompt += "\nPrevious guesses and their errors:\n"
        for i, (A_i, B_i) in enumerate(previous_guesses):
            error_i = previous_errors[i]
            prompt += f"Iteration {i+1}: A = {A_i:.6f}, B = {B_i:.6f}, Error = {error_i:.6f}\n"
        prompt += "\nBased on the previous attempts, provide new estimates for A and B.\n"

    prompt += "Please explain your reasoning and then provide your final answer in the following format:\nA = [value]\nB = [value]\n"

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

def update_plot(x, y, previous_guesses, iter):
    """
    Update the plot by adding the new model curve.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data Points')
    x_fit = np.linspace(x.min(), x.max(), 100)
    for i, (A_i, B_i) in enumerate(previous_guesses):
        y_fit = A_i * x_fit**2 + B_i
        label = f'Iteration {i+1}'
        plt.plot(x_fit, y_fit, label=label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data and Model Fits')
    plt.legend()
    # Save the plot image (optional, for record-keeping)
    plt.savefig(f'plot_iteration_{iter}.png')
    plt.close()

# Initialize lists to store previous guesses and errors
previous_guesses = []
previous_errors = []
data_record = []

# Iterative loop
for iteration in range(10):
    print(f"Iteration {iteration+1}:")

    # Generate the plot description
    description = generate_plot_description(x, y)

    # Update the plot with previous guesses (for record-keeping)
    update_plot(x, y, previous_guesses, iteration)

    # Prepare the prompt for the LLM
    prompt = prepare_prompt(description, previous_guesses, previous_errors)

    # Call the OpenAI API
    try:
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "user", "content": prompt}
        #     ],
        #     max_tokens=500,
        #     temperature=0.7,
        # )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
            stream=True
        )

        assistant_reply = "".join([str(chunk.choices[0].delta.content) for chunk in completion])

    except Exception as e:
        print(f"An error occurred during API call: {e}")
        break

    # Get the assistant's reply
    # assistant_reply = response['choices'][0]['message']['content']
    print("Assistant's response:")
    print(assistant_reply)
    print("--------------------------------------------------------")
    # Extract A and B from the assistant's reply
    A_guess, B_guess = extract_values(assistant_reply)
    print(A_guess, B_guess)
    print("--------------------------------------------------------")

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
        'assistant_reply': assistant_reply
    }
    data_record.append(record)

    # Print the results of the current iteration
    print(f"Estimated A = {A_guess:.6f}, B = {B_guess:.6f}, Error = {error:.6f}\n")

# Save the data records to a JSON file
with open('data_record.json', 'w') as f:
    json.dump(data_record, f, indent=4)