"""
files.py

Utility functions for finding and working with files.
"""

import os
from typing import List, Dict, Union, Optional
import pandas as pd
from provada.paths import REPO_ROOT


def get_filepath_from_name_or_path(
    file_path_or_name: str,
    file_extensions: Union[str, List[str]] = None,
):
    """
    Finds the unique file specified by the file path or name.

    If file_path_or_name is a path to a file that exists, that file is returned.
    Otherwise, the function will search for files within the root directory that
    match the name of the file.

    Args:
        file_path_or_name (str): The path to the file or the name of the file.
        file_extensions (Union[str, List[str]]): The file extensions to search for.
            If None, all files are searched for.

    Returns:
        str: The path to the file.
    """

    # If the file path or name is a path to a file that exists, return it
    if os.path.exists(file_path_or_name):
        all_collected_files = collect_filepaths(
            os.path.dirname(file_path_or_name), [os.path.splitext(file_path_or_name)[1]]
        )
    else:
        # Collect all filepaths in the root directory with the specified extensions
        all_collected_files = collect_filepaths(str(REPO_ROOT), file_extensions)

    # Initialize the component file
    collected_file = None

    # Determine if we have an absolute path or relative path match
    for current_file in all_collected_files:
        if file_path_or_name in [
            current_file["absolute_path"],
            current_file["relative_path"],
        ]:
            if collected_file is not None:
                raise ValueError(
                    f"Multiple component files found for {file_path_or_name}"
                    f"\n\t{collected_file['absolute_path']}"
                    f"\n\t{current_file['absolute_path']}"
                )
            collected_file = current_file

    # If we don't have a path match, check for a filename match
    if collected_file is None:
        for current_file in all_collected_files:
            if file_path_or_name in [
                current_file["filename"],
                current_file["basename"],
            ]:
                if collected_file is not None:
                    raise ValueError(
                        f"Multiple component files found for {file_path_or_name}"
                        f"\n\t{collected_file['absolute_path']}"
                        f"\n\t{current_file['absolute_path']}"
                    )
                collected_file = current_file

    # If we don't have a match, raise an error
    if collected_file is None:
        raise ValueError(f"No file found for {file_path_or_name}")

    return collected_file


def collect_filepaths(
    root_dir: str, file_extensions: Union[str, List[str]]
) -> List[Dict[str, str]]:
    """
    Collects files and paths of files with the specified extension types in the
    specified directory and it's sub-directories.

    Parameters:
        root_dir (str): The path to the directory to collect files from.
        file_extensions (List[str]): The list of file extensions to collect.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the
            absolute path, basename, filename, extension, and relative path of each file.
    """

    if isinstance(file_extensions, str):
        file_extensions = [file_extensions]

    for i, file_extension in enumerate(file_extensions):
        if not file_extension.startswith("."):
            file_extensions[i] = "." + file_extension

    collected_files = []

    # Walk through the source directory
    for root, dirs, files in os.walk(root_dir):
        # For each file
        for file in files:
            for ext in file_extensions:
                if file.endswith(ext):
                    # Add the current file
                    collected_files.append(
                        {
                            "absolute_path": os.path.join(root, file),
                            "basename": file,
                            "filename": file[: len(file) - len(ext)],
                            "ext": ext,
                            "relative_path": os.path.join(root, file).replace(
                                root_dir, ""
                            )[1:],
                            "relative_depth": len(
                                os.path.join(root, file)
                                .replace(root_dir, "")
                                .split(os.sep)
                            ),
                        }
                    )

    return collected_files


def dataframe_to_fasta(
    df: pd.DataFrame,
    output_fasta_path: str,
    sequence_column_name: str = "sequence",
    id_column_name: Optional[str] = None,
    header_columns: Optional[List[str]] = None,
):
    """
    Creates a fasta file from a dataframe.

    The fasta file will have the following format:
    ```
    >id | header_columns[0] | header_columns[1] | ...
    sequence
    ```

    Args:
        df (pd.DataFrame): The dataframe to convert to a fasta file
        output_fasta_path (str): The path to the output fasta file
        sequence_column_name (str): The name of the column containing the sequences
        id_column_name (str): The name of the column containing the id
        header_columns (List[str]): The columns to include in the header of the fasta file
            deliminted by a pipe (`|`)
    """

    # Ensure sequence_column_name is in the dataframe
    if sequence_column_name not in df.columns:
        raise ValueError(f"{sequence_column_name} is not in dataframe")

    if header_columns is None:
        header_columns = []

    # Ensure all header columns are in the dataframe
    for header_column in header_columns:
        if header_column not in df.columns:
            raise ValueError(f"{header_column} is not in dataframe")

    # If id_column_name is not specified, use the index as the id
    if id_column_name is None:
        id_column_name = "id"
        df = df.copy()
        df[id_column_name] = df.index

    # Write the fasta file
    with open(output_fasta_path, "w") as f:
        for i, seq in df.iterrows():
            write_str = f">{seq[id_column_name]}"

            # Add the header columns
            for header_column in header_columns:
                write_str += f"|{seq[header_column]}"

            # Finally add the sequence
            write_str += f"\n{seq[sequence_column_name]}\n"

            f.write(write_str)


def fasta_to_dataframe(
    input_fasta_path: str,
    sequence_column_name: str = "sequence",
    id_column_name: str = "id",
    header_delimiter: str = "|",
    header_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Creates a dataframe from a fasta file.

    Converts a fasta file with headers in the format:
    ```
    >id | header_columns[0] | header_columns[1] | ...
    sequence
    ```

    Args:
        input_fasta_path (str): The path to the input fasta file
        sequence_column_name (str): The name of the column to store sequences
        id_column_name (str): The name of the column to store the IDs
        header_delimiter (str): The delimiter used between fields in the header
        header_columns (List[str]): The names to assign to additional columns from the header.
            If None, auto-generates column names as "field1", "field2", etc.

    Returns:
        pd.DataFrame: DataFrame containing sequences and header information
    """

    if not os.path.exists(input_fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {input_fasta_path}")

    # Initialize lists to store data
    ids = []
    additional_fields = []
    sequences = []

    with open(input_fasta_path, "r") as f:
        current_id = None
        current_fields = []
        current_sequence = []

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                # If we were processing a sequence, save it
                if current_id is not None:
                    ids.append(current_id)
                    additional_fields.append(current_fields)
                    sequences.append("".join(current_sequence))

                # Start new sequence
                header_parts = line[1:].split(header_delimiter)
                current_id = header_parts[0].strip()
                current_fields = [field.strip() for field in header_parts[1:]]
                current_sequence = []
            else:
                # Accumulate sequence
                current_sequence.append(line)

        # Add the last sequence if it exists
        if current_id is not None:
            ids.append(current_id)
            additional_fields.append(current_fields)
            sequences.append("".join(current_sequence))

    # Determine number of additional fields and their column names
    max_fields = (
        max(len(fields) for fields in additional_fields) if additional_fields else 0
    )

    if header_columns is None and max_fields > 0:
        header_columns = [f"field{i+1}" for i in range(max_fields)]

    # Create dictionary for dataframe
    data = {id_column_name: ids, sequence_column_name: sequences}

    # Add additional fields
    for i in range(max_fields):
        col_name = header_columns[i] if i < len(header_columns) else f"field{i+1}"
        data[col_name] = [
            fields[i] if i < len(fields) else None for fields in additional_fields
        ]

    # Create and return dataframe
    return pd.DataFrame(data)
