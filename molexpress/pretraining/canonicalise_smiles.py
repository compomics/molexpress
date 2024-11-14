import multiprocessing as mp
from rdkit import Chem
from tqdm import tqdm
import os

# Function to canonicalize SMILES
def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to molecule object
    if mol:  # Check if molecule conversion was successful
        return Chem.MolToSmiles(mol, canonical=True)  # Return canonical SMILES
    else:
        return None

# Process a chunk of SMILES
def process_chunk(smiles_chunk):
    valid_smiles = []
    invalid_smiles = []
    for smiles in smiles_chunk:
        canonical_smiles = canonicalize_smiles(smiles.strip())
        if canonical_smiles:
            valid_smiles.append(canonical_smiles)
        else:
            invalid_smiles.append(smiles.strip())
    return valid_smiles, invalid_smiles

# Read SMILES from input file and split them into chunks
def process_smiles_file(input_file, output_file, invalid_file, num_processes=4, chunk_size=100000):
    # Get the total number of lines (SMILES strings)
    total_lines = sum(1 for _ in open(input_file, 'r'))

    # Use multiprocessing to process the file in parallel
    with open(input_file, 'r') as infile:
        smiles_list = infile.readlines()
    
    # Split the smiles into chunks
    smiles_chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]

    # Set up a multiprocessing pool
    with mp.Pool(processes=num_processes) as pool:
        # Process each chunk in parallel
        results = list(tqdm(pool.imap(process_chunk, smiles_chunks), total=len(smiles_chunks)))

    # Gather results
    valid_smiles = []
    invalid_smiles = []
    for valid, invalid in results:
        valid_smiles.extend(valid)
        invalid_smiles.extend(invalid)

    # Write the results to the output files
    with open(output_file, 'w') as outfile:
        outfile.write('\n'.join(valid_smiles) + '\n')
    with open(invalid_file, 'w') as invalid_outfile:
        invalid_outfile.write('\n'.join(invalid_smiles) + '\n')

# Example usage
input_file = 'filtered_pubchem.txt'       # Your input file containing SMILES strings
output_file = 'canon_filtered_pubchem.txt'  # Output file for valid canonical SMILES
invalid_file = 'invalid_smiles.txt'   # Output file for invalid SMILES

# Adjust num_processes based on your machine's CPU cores, and tune chunk_size based on file size
process_smiles_file(input_file, output_file, invalid_file, num_processes=8, chunk_size=100000)
