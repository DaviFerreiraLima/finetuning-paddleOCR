

# Fine-Tuning do PaddleOCR para Reconhecimento de Placas Veiculares BR  

## 📋 Visão Geral  

Este projeto visa adaptar o **PaddleOCR** (um sistema avançado de **OCR - Optical Character Recognition**) para reconhecer com alta precisão **placas de veículos brasileiras**, incluindo os formatos **Mercosul** e **modelos antigos**.  

🔹 O **PaddleOCR** é um modelo pré-treinado para reconhecimento de **texto genérico**, mas não está otimizado para placas veiculares, que possuem características distintas, como:  
✔ Caracteres específicos (**A-Z, 0-9, hífen**)  
✔ Fontes padronizadas  
✔ Fundos variados e ruídos  
✔ Condições de captura desfavoráveis  

---

## ❓ Por Que Este Projeto É Necessário?  

### 🎯 Especialização do OCR para Placas  
🔹 Modelos genéricos de OCR **falham** na leitura de placas devido à sua estrutura única.  
🔹 O **fine-tuning** melhora a acurácia em caracteres semelhantes como **"0" vs "O"**, **"5" vs "S"**, etc.  

### 🇧🇷 Adaptação ao Contexto Brasileiro  
🔹 Suporte aos padrões **Mercosul** (**ABC1D23**) e antigos (**ABC-1234**).  
🔹 Lida com variações de **iluminação, sujeira e angulação**.  

### ⚡ Pronto para Produção  
O modelo final pode ser integrado em:  
✔ **Sistemas de monitoramento de tráfego**  
✔ **Estacionamentos automatizados**  
✔ **Aplicações de fiscalização**  

### 💻 Multiplataforma  
✔ **Windows (GPU NVIDIA)**  
✔ **macOS (CPU)**  
✔ **Linux**  

---

## Pré-requisitos Técnicos

### 1 Download do Modelo Base
Antes de iniciar, baixe o modelo pré-treinado:

```bash
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar -P dataset/
```

### 2. Preparação das Imagens
- Todas as imagens devem estar na pasta `dataset/filter_images/`

### 3. Verificação do labels.json
O arquivo deve conter um array de objetos no formato:
```json
[
  {
    "text": "ABC1D23",
    "image_path": "filter_images/placa_001.jpg"
  },
  {
    "text": "XYZ9W87", 
    "image_path": "filter_images/placa_002.jpg"
  }
```
---

## 📌 Estrutura do Projeto  

```bash
📂 paddleocr-finetune-br
├── 📂 dataset/                # Dados de treino e validação
│   ├── 📂 train/              # Dados de treinamento
│   ├── 📂 eval/               # Dados de validação
│   ├── 📄 caracteres.txt      # Lista de caracteres válidos
│   ├── 📄 train.txt           # Labels do conjunto de treinamento
│   ├── 📄 eval.txt            # Labels do conjunto de validação
├── 📂 configs/                # Configurações do treinamento
│   ├── 📄 config_finetune.yaml
├── 📂 scripts/                # Scripts úteis
│   ├── 📄 prepare_data.py     # Prepara os dados
│   ├── 📄 train.py            # Executa o fine-tuning
│   ├── 📄 export_model.py     # Exporta o modelo treinado
└── 📂 output/                 # Modelos treinados e logs
```

---

## 🛠️ Configuração do Ambiente  

### 📌 Requisitos Gerais  
✔ Python **3.8+**  
✔ 8GB+ **RAM** (**16GB recomendado**)  
✔ 10GB+ de **espaço em disco**  
✔ **Conda** (Miniconda ou Anaconda)  

### 🎮 Requisitos para GPU  
| Componente       | Especificação                          |  
|------------------|--------------------------------------|  
| Placa de Vídeo   | NVIDIA GPU (Compute Capability 3.5+) |  
| Drivers NVIDIA   | Versão mais recente                  |  
| CUDA Toolkit     | 11.2-11.7 (compatível com Paddle 2.6) |  
| cuDNN           | 8.1.1+                               |  

### 🖥️ Compatibilidade de Sistemas  
| SO       | CPU  | GPU  | Observações                     |  
|----------|------|------|---------------------------------|  
| Windows  | ✅   | ✅   | Melhor performance com GPU      |  
| Linux    | ✅   | ✅   | Suporte completo                |  
| macOS    | ✅   | ❌   | Somente CPU (M1/M2 compatível)  |  

---

## 🔧 Instalação  

### 🔹 Windows (GPU NVIDIA)  
```bash
conda create -n paddle_env python=3.8 -y
conda activate paddle_env
pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/windows/mkl.html
```

### 🔹 macOS (CPU)  
```bash
conda create -n paddle_env python=3.8 -y
conda activate paddle_env
pip install paddlepaddle==2.6.1
```

### 🔹 Linux (CPU/GPU)  
```bash
# Para GPU
pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl.html

# Para CPU
pip install paddlepaddle==2.6.1
```

---

### Setup adicional do ambiente e preparação dos dados

```python
python setup_dados_paddle_ocr.py
```

## 📊 Treinamento  

### 🔹 Configuração (`config_finetune.yaml`)  
```yaml
Global:
  use_gpu: true
  epoch_num: 300
  batch_size_per_card: 64
  pretrained_model: ./dataset/pretrain_models/en_PP-OCRv3_rec_train/best_accuracy
  character_dict_path: ./dataset/caracteres.txt
  max_text_length: 8
```

### 🔹 Iniciando o Treinamento

#### Para GPU
```bash
python PaddleOCR/tools/train.py -c config_finetune_com_gpu.yaml
```
#### Para CPU
```bash
python PaddleOCR/tools/train.py -c config_finetune.yaml
```

---

## 📤 Exportação do Modelo  

Substitua iter_epoch_XXX pela melhor epoch observada:
```bash
python PaddleOCR/tools/export_model.py \
    -c configs/config_finetune.yaml \
    -o Global.pretrained_model=./output/placa_br_model/iter_epoch_100 \
    Global.save_inference_dir=./inference_model
```

---

## 🚀 Testando o Modelo Exportado  

```bash
python test.py \
    --image_path sua_imagem.jpg \
    --rec_model_dir inference_model \
    --use_gpu False  # Mude para True se usando GPU
```

## ❌ Solução de Problemas  

| Erro/Sintoma       | Causa Provável            | Solução                          |  
|--------------------|--------------------------|----------------------------------|  
| CUDA out of memory | `batch_size` muito alto  | Reduza o `batch_size` em 50%    |  
| NaN in loss       | Taxa de aprendizado alta  | Reduzir `lr` para `0.0001`      |  

---

## 📌 Considerações Finais  

✔ Para um bom desempenho, recomenda-se um **dataset com pelo menos 5.000 imagens**.  
✔ O tempo de treinamento varia de **2-12 horas**, dependendo do hardware.  
✔ O modelo pode ser integrado via **Python, C++ ou API REST**.  

```

