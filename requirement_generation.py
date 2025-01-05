import importlib.metadata

def create_requirements_file(library_names, output_file='requirements.txt'):
    """
    Generates a requirements.txt file with the installed versions of the specified libraries.
    
    Args:
        library_names (list): A list of library names to include in the requirements file.
        output_file (str): The name of the output file. Default is 'requirements.txt'.
    """
    requirements = []
    for lib in library_names:
        try:
            # Get the library's installed version
            version = importlib.metadata.version(lib)
            requirements.append(f"{lib}=={version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"Warning: {lib} is not installed on your system.")
    
    # Write the requirements to the file
    with open(output_file, 'w') as file:
        file.write("\n".join(requirements))
    
    print(f"Requirements file '{output_file}' has been created successfully!")

# Example usage:
library_list = ['arize-phoenix', 'arize-phoenix-otel', 'openinference-instrumentation-groq', 'groq', 'sentence-transformers', 'pandas', 'python-dotenv', 'streamlit','plotly','litellm']  # Replace with your libraries
create_requirements_file(library_list)
