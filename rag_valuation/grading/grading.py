import re
import json
import pandas as pd

def evaluate_ai_response(correct_label, generated_label):
    """
    Evaluate the AI generated label based on the correct_label.
    
    Args:
    correct_label (str): The correct label for the response.
    generated_label (str): The label generated by the AI.
    
    Returns:
    str: 'correct', 'incorrect', or 'not valid' based on the evaluation.
    """

    def normalize_label(label):
        # Remove leading/trailing spaces and convert to uppercase
        label = label.strip().upper()
        # Split the label into parts (e.g., "B" and "NO")
        parts = re.split(r'\W+', label)
        return parts

    def contains_valid_answer(label):
        # Check if the label contains a valid answer format
        return bool(re.search(r'\b(A|B)\b|\b(CORRECT|MISINFORMATION)\b', label.upper()))

    # Normalize the generated label
    generated_label_normalized = ' '.join(normalize_label(generated_label)).upper()

    # If the generated label does not contain a valid answer format, return 'not valid'
    if not contains_valid_answer(generated_label):
        return 'invalid'

    # Normalize the correct label and check if it is in the generated label
    correct_label_parts = normalize_label(correct_label)
    if all(part in generated_label_normalized for part in correct_label_parts):
        return 'correct'
    else:
        return 'incorrect'

def process_csv(generated_answers, correct_labels):
    # Load CSV into DataFrame
    df = pd.read_csv(generated_answers)

    # Apply the evaluation function to each row using corresponding correct label
    df['eval'] = df.apply(lambda row: evaluate_ai_response(correct_labels[row.name], row['response']), axis=1)

    return df

def run(generated_answers, correct_answers, output_path):
    # Read the correct answers file and parse JSON lines
    correct_labels = []
    with open(correct_answers, 'r') as f:
        for line in f:
            json_obj = json.loads(line)
            correct_label = f"({json_obj['correct_choice_letter']}) {json_obj['correct_choice_text']}"
            correct_labels.append(correct_label)

    # Process the generated answers with the correct labels
    processed_df = process_csv(generated_answers, correct_labels)

    # Save the processed data to the output path
    processed_df.to_csv(output_path, index=False)

    eval_stats = processed_df['eval'].value_counts().to_dict()

    # Log the statistics in a structured way
    print("Evaluation Statistics:")
    print(f"Correct: {eval_stats.get('correct', 0)}")
    print(f"Incorrect: {eval_stats.get('incorrect', 0)}")
    print(f"Invalid: {eval_stats.get('invalid', 0)}")

    return eval_stats


def run_rag_grades(generated_answers, correct_answers, output_path):
    # Read the correct answers file and parse JSON lines
    correct_labels = {}
    with open(correct_answers, 'r') as f:
        for line in f:
            json_obj = json.loads(line)
            correct_label = f"({json_obj['correct_choice_letter']}) {json_obj['correct_choice_text']}"
            correct_labels[json_obj['id']] = correct_label

    # Load generated answers CSV into DataFrame
    df = pd.read_csv(generated_answers)

    # Initialize a dictionary to keep track of rag grades
    rag_grades = {}

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        question_id = row['question_id']
        rag_id = row['rag_id']
        response = row['response']

        # Evaluate the response
        eval_result = evaluate_ai_response(correct_labels[question_id], response)

        # Update rag_grades dictionary
        if rag_id not in rag_grades:
            rag_grades[rag_id] = {'overall_correct': 0, 'questions_correct': [],
                                  'overall_incorrect': 0, 'questions_incorrect': []}

        if eval_result == 'correct':
            rag_grades[rag_id]['overall_correct'] += 1
            rag_grades[rag_id]['questions_correct'].append(question_id)
        elif eval_result == 'incorrect':
            rag_grades[rag_id]['overall_incorrect'] += 1
            rag_grades[rag_id]['questions_incorrect'].append(question_id)

    # Save the rag_grades dictionary to the output path
    with open(output_path, 'w') as f:
        json.dump(rag_grades, f, indent=4)

    return rag_grades
