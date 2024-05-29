import csv

# Open the input file for reading with 'utf-8' encoding
with open('Finance_Related_Articles_CSV_TRUE.csv', 'r', encoding='utf-8') as input_file:
    # Create a CSV reader object
    csv_reader = csv.reader(input_file)
    # Read the header
    header = next(csv_reader)
    # Find the index of the 'text' column
    text_index = header.index('text')
    # Read and process each row
    rows = [row[text_index] for row in csv_reader]

# Write the cleaned data back to a new file
with open('Finance_TRUE.csv', 'w', newline='', encoding='utf-8') as output_file:
    # Create a CSV writer object
    csv_writer = csv.writer(output_file)
    # Write the header
    csv_writer.writerow(['text'])
    # Write each row
    for row in rows:
        csv_writer.writerow([row])

