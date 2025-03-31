import json
import os
import random
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

# Configurations
BASE_DIR = "dataset"
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
FINETUNE_DIR = os.path.join(BASE_DIR, "finetuning")
FILTER_IMAGES_DIR = os.path.join(BASE_DIR, "filter_images")
MODEL_TAR_PATH = os.path.join(BASE_DIR, "en_PP-OCRv3_rec_train.tar")
MODEL_DIR = os.path.join(BASE_DIR, "pretrain_models")
PADDLEOCR_REPO = "https://github.com/PaddlePaddle/PaddleOCR.git"
PADDLEOCR_DIR = "PaddleOCR"

def setup_environment():
    """Install dependencies and setup environment"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    os.makedirs(FINETUNE_DIR, exist_ok=True)
    os.makedirs(FILTER_IMAGES_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Detect operating system
    system = platform.system().lower()
    is_windows = system == 'windows'
    is_mac = system == 'darwin'
    
    # Install paddlepaddle based on OS
    if is_windows:
        print("⏳ Installing PaddlePaddle-GPU 2.6.1 via Conda for Windows...")
        subprocess.run([
            "conda", "install", "paddlepaddle-gpu==2.6.1", "cudatoolkit=11.7",
            "--channel", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/",
            "--channel", "conda-forge", "-y"
        ], check=True)
    elif is_mac:
        print("⏳ Installing PaddlePaddle 2.6.1 via Conda for Mac...")
        subprocess.run([
            "conda", "install", "paddlepaddle==2.6.1",
            "--channel", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/",
            "-y"
        ], check=True)
    else:
        print("⏳ Installing PaddlePaddle 2.6.1 via Conda for Linux...")
        subprocess.run([
            "conda", "install", "paddlepaddle==2.6.1",
            "--channel", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/",
            "-y"
        ], check=True)
    
    # Clone PaddleOCR if it doesn't exist
    if not os.path.exists(PADDLEOCR_DIR):
        print("⏳ Cloning PaddleOCR repository...")
        subprocess.run(["git", "clone", PADDLEOCR_REPO], check=True)
    
    # Install PaddleOCR requirements
    print("⏳ Installing PaddleOCR dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", 
        os.path.join(PADDLEOCR_DIR, "requirements.txt")
    ], check=True)
    
    # Extrai o modelo se existir (versão simplificada)
    if os.path.exists(MODEL_TAR_PATH):
        print("⏳ Extraindo modelo pré-treinado...")
        with tarfile.open(MODEL_TAR_PATH, 'r') as tar:
            tar.extractall(path=MODEL_DIR)
        print(f"✓ Modelo extraído em: {os.path.abspath(MODEL_DIR)}")
    else:
        print(f"⚠️ Arquivo do modelo não encontrado em: {os.path.abspath(MODEL_TAR_PATH)}")
        print("Por favor, coloque o modelo en_PP-OCRv3_rec_train.tar na raiz do projeto")
        print("Você pode baixá-lo em:")
        print("https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar")

def process_item(item):
    """Process each JSON item with robust path verification"""
    if 'text' not in item or 'image_path' not in item:
        print(f"Invalid item: {item}")
        return None

    text = item['text']
    original_path = item['image_path']
    image_name = os.path.basename(original_path)
    
    # Determine the correct path in filter_images
    if 'antigas' in original_path:
        new_path = os.path.join(FILTER_IMAGES_DIR, f'antigas_degrade_{image_name}')
    elif 'mercosul' in original_path:
        new_path = os.path.join(FILTER_IMAGES_DIR, f'mercosul_degrade_{image_name}')
    else:
        new_path = os.path.join(FILTER_IMAGES_DIR, image_name)

    # Verify the image exists in filter_images
    if not os.path.exists(new_path):
        print(f"Image not found in filter_images: {new_path}")
        return None

    return {
        'path_da_imagem': new_path,
        'textDaPlaca': text
    }

def generate_characters_file():
    """Generate file with valid characters"""
    caracteres_placas = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '-'
    ]
    
    output_path = os.path.join(FINETUNE_DIR, 'caracteres_placas_br.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for char in caracteres_placas:
            f.write(f"{char}\n")
    
    print(f"✓ Character file generated: {output_path}")

def generate_csv_files(data):
    """Generate CSV files only with existing images"""
    # Process and filter data
    processed_data = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_item, data)
        processed_data = [item for item in results if item is not None]
    
    if not processed_data:
        raise Exception("No valid images found in filter_images!")
    
    # Split data
    random.seed(42)
    train_data, temp_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    test_data, eval_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Generate CSVs
    csv_files = {
        'train.csv': train_data,
        'test.csv': test_data,
        'eval.csv': eval_data
    }
    
    for filename, data in csv_files.items():
        csv_path = os.path.join(FINETUNE_DIR, filename)
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            for row in data:
                abs_path = os.path.abspath(row['path_da_imagem'])
                csvfile.write(f"{abs_path}, {row['textDaPlaca']}\n")
        #        relative_path = os.path.join('.', row['path_da_imagem'])
        #        csvfile.write(f"{relative_path}, {row['textDaPlaca']}\n")
       #         csvfile.write(f"{row['path_da_imagem']}, {row['textDaPlaca']}\n")
        print(f"✓ CSV generated: {csv_path}")
        
        # Convert to PaddleOCR format
        txt_path = os.path.join(FINETUNE_DIR, filename.replace('.csv', '.txt'))
        gen_label_script = os.path.join(PADDLEOCR_DIR, "ppocr", "utils", "gen_label.py")
        
        subprocess.run([
            sys.executable,
            gen_label_script,
            '--mode=rec',
            f'--input_path={csv_path}',
            f'--output_label={txt_path}'
        ], check=True)
        
        print(f"✓ PaddleOCR file generated: {txt_path}")

def main():
    try:
        # Setup environment
        setup_environment()
        
        # Load data
        print(f"\n📂 Reading label file: {os.path.abspath(LABELS_PATH)}")
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Generate files
        print("\n🚀 Generating training files...")
        generate_csv_files(data)
        generate_characters_file()
        
        print("\n✅ Process completed successfully!")
        print(f"📂 Output folder: {os.path.abspath(FINETUNE_DIR)}")
        print(f"🖼️ Processed images: {os.path.abspath(FILTER_IMAGES_DIR)}")
        print(f"🤖 Pretrained model: {os.path.abspath(MODEL_DIR)}")
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command execution error: {e.cmd}")
        print(f"Error code: {e.returncode}")
        if e.output:
            print(f"Output: {e.output}")
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {str(e)}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()