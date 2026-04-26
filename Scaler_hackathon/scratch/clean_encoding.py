import os

def clean_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    with open(filepath, 'rb') as f:
        content = f.read()
    
    # Decodes as utf-8, then encodes as ascii ignoring errors
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        text = content.decode('latin-1')
        
    cleaned_text = text.encode('ascii', 'ignore').decode('ascii')
    
    with open(filepath, 'w', encoding='ascii') as f:
        f.write(cleaned_text)
    print(f"Cleaned {filepath}")

files_to_clean = [
    "Blog.md",
    "README.md",
    "openenv.yaml",
    "server/app.py",
    "models.py",
    "data_generation.py",
    "server/baseline.py"
]

for f in files_to_clean:
    clean_file(f)
