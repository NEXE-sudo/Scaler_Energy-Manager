import os

def clean_all_files(root_dir):
    exclude_dirs = {'.venv', '.git', 'node_modules', '__pycache__', 'agent-grid-view/node_modules'}
    
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith(('.py', '.md', '.yaml', '.yml', '.txt', '.json', '.jsonl', '.tsx', '.ts', '.html', '.css', 'Dockerfile')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'rb') as f:
                        content = f.read()
                    
                    try:
                        text = content.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            text = content.decode('latin-1')
                        except:
                            print(f"Skipping binary or unknown encoding: {filepath}")
                            continue
                            
                    cleaned_text = text.encode('ascii', 'ignore').decode('ascii')
                    
                    if text != cleaned_text:
                        with open(filepath, 'w', encoding='ascii') as f:
                            f.write(cleaned_text)
                        print(f"Cleaned: {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

clean_all_files(".")
