import csv

def convert_to_csv(input_file, output_file):
    """
    Convert a text file to CSV format.

    Args:
    input_file (str): The path to the input text file.
    output_file (str): The path where the output CSV file will be saved.

    This function reads the input file line by line, processes each line to extract
    the required fields, and writes them into a CSV file with the specified column headers.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        # Define CSV writer with the required column headers
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['question_id', 'rag_id', 'response'])

        for line in infile:
            # Strip whitespace and the enclosing square brackets
            line = line.strip()[1:-1]

            # Split the line into components
            components = line.split(',', 2)  # Split into 3 parts

            if len(components) != 3:
                continue  # Skip lines that do not have exactly 3 components

            # Extract and clean each component
            question_id = components[0].strip()
            rag_id = components[1].strip() if components[1].strip() != 'None' else 'None'
            response = components[2].strip()

            # Remove any commas and escaped quotes from the response
            response = response.replace(',', '').replace('\\"', '"')

            # Write the row to the CSV file
            csv_writer.writerow([question_id, rag_id, response])

# Example usage
convert_to_csv('climate_fever_baseline_responses.txt', 'climate_fever_baseline_responses.csv')
