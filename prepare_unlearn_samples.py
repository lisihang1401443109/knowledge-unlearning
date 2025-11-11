import pandas as pd
import os
import random

def prepare_unlearn_samples():
    """
    Generates CSV files with a small sample from each domain for unlearning experiments.
    """
    source_dir = 'data/domain_main'
    output_dir = 'data/unlearn_samples'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    domains = [
        'books3',
        'enron_emails',
        'freelaw',
        'github',
        'license',
        'pile-cc',
        'pubmed_central',
        'uspto_backgrounds'
    ]

    for domain in domains:
        print(f"Processing domain: {domain}")
        
        source_file_path = os.path.join(source_dir, f'{domain}_8_0.csv')
        
        if not os.path.exists(source_file_path):
            print(f"Warning: Source file not found for domain '{domain}': {source_file_path}")
            continue

        # Read the source csv
        source_df = pd.read_csv(source_file_path, lineterminator='\n')
        
        # Take 4 random samples
        if len(source_df) > 4:
            sampled_df = source_df.sample(n=4, random_state=42) # use random_state for reproducibility
        else:
            sampled_df = source_df

        # Create the new dataframe in the desired format
        new_df = pd.DataFrame({
            'doc_id': range(len(sampled_df)),
            'corpus': domain,
            'text': sampled_df['text']
        })

        output_filename = f'unlearn_sample_{domain}.csv'
        output_path = os.path.join(output_dir, output_filename)
        
        new_df.to_csv(output_path, index=False)
        print(f"Saved sampled data to {output_path}")

if __name__ == '__main__':
    prepare_unlearn_samples()
