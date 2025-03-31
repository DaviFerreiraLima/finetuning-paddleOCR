import json
import os
import random
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import platform

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
            "conda", "install", "paddlepaddle==2.6.1", "cudatoolkit=11.7",
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
    """Generate CSV files that will be converted to PaddleOCR format"""
    # Process and filter data
    processed_data = []
    for item in data:
        processed_item = process_item(item)
        if processed_item:
            processed_data.append(processed_item)
    
    # Split data
    random.seed(42)
    train_data, temp_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    test_data, eval_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Generate CSVs with proper formatting
    csv_files = {
        'train.csv': train_data,
        'test.csv': test_data,
        'eval.csv': eval_data
    }
    
    for filename, dataset in csv_files.items():
        csv_path = os.path.join(FINETUNE_DIR, filename)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            for item in dataset:
                abs_path = os.path.abspath(item['path_da_imagem'])
                
                # Critical Windows fixes:
                if platform.system() == 'Windows':
                    # Normalize path and handle spaces
                    abs_path = os.path.normpath(abs_path)
                    if ' ' in abs_path:
                        abs_path = f'"{abs_path}"'
                
                # Write with TAB separator (not space)
                csvfile.write(f"{abs_path}\t{item['textDaPlaca']}\n")
        
        print(f"✓ CSV file generated: {csv_path}")
        
        # Convert to PaddleOCR format
        txt_path = os.path.join(FINETUNE_DIR, filename.replace('.csv', '.txt'))
        gen_label_script = os.path.join(PADDLEOCR_DIR, "ppocr", "utils", "gen_label.py")
        
        # Run PaddleOCR's label generator with explicit encoding
        try:
            subprocess.run([
                sys.executable,
                gen_label_script,
                '--mode=rec',
                f'--input_path={csv_path}',
                f'--output_label={txt_path}',
                '--delimiter="\t"'  # Explicitly specify we used tabs
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"✓ PaddleOCR label file generated: {txt_path}")
            
            # Verify the generated file
            with open(txt_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if not first_line or '\t' not in first_line:
                    print(f"⚠️ Warning: Generated file might have formatting issues in first line: {first_line}")
                    
        except subprocess.CalledProcessError as e:
            print(f"❌ Error generating PaddleOCR file for {csv_path}")
            print(f"Command failed: {e.cmd}")
            print(f"Error output: {e.stderr.decode('utf-8') if e.stderr else 'None'}")
            
            # Fallback: manual conversion if PaddleOCR fails
            print("Attempting manual conversion...")
            try:
                with open(csv_path, 'r', encoding='utf-8') as f_in, \
                     open(txt_path, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        f_out.write(line)
                print(f"✓ Manual conversion successful: {txt_path}")
            except Exception as fallback_error:
                print(f"❌ Manual conversion failed: {str(fallback_error)}")

def main():
    try:
        # Setup environment
        setup_environment()
        
        # Load data
        print(f"\n📂 Reading label file: {os.path.abspath(LABELS_PATH)}")
        try:
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            # Tentar outras codificações se UTF-8 falhar
            try:
                with open(LABELS_PATH, 'r', encoding='latin-1') as f:
                    data = json.load(f)
            except Exception as e:
                raise Exception(f"Failed to read label file with both UTF-8 and latin-1 encodings: {str(e)}")
        
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
            try:
                print(f"Output: {e.output.decode('utf-8')}")
            except:
                print(f"Output: {str(e.output)}")
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {str(e)}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
