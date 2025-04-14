# NinjaNLP - Naruto Anime Analysis Tools

## About

NinjaNLP is an innovative NLP toolkit specifically designed for analyzing the Naruto anime series. By leveraging state-of-the-art language models and machine learning techniques, this project offers unique insights into the rich narrative world of Naruto through:

- **Advanced Theme Analysis**: Automatically identify and track themes like friendship, perseverance, and conflict across episodes using BART zero-shot classification
- **Character Relationship Mapping**: Visualize the complex web of character interactions and relationships using Named Entity Recognition
- **Jutsu Classification System**: Categorize and analyze different ninja techniques (jutsu) mentioned in the series using DistilBERT
- **Interactive Naruto Chatbot**: Engage with a character-accurate chatbot trained on Naruto's dialogue using LLaMA model

Perfect for:
- Anime researchers and content creators
- Naruto fans seeking deeper analysis
- NLP practitioners interested in anime/manga domain applications
- Students learning about applied machine learning in entertainment analysis

Built with modern NLP technologies including Hugging Face Transformers, SpaCy, and Gradio, NinjaNLP demonstrates practical applications of machine learning in media analysis while providing valuable insights into one of the world's most popular anime series.

## 🚀 Key Features

- **Theme Classification**: Zero-shot classification of themes in episodes using BART
- **Character Network Analysis**: Visualization of character relationships using Named Entity Recognition
- **Jutsu Classification**: Classification of jutsu techniques using DistilBERT
- **Character Chatbot**: Naruto-themed chatbot using LLaMA model

## 🛠️ Models Used

- Theme Classification: `facebook/bart-large-mnli`
- Named Entity Recognition: `en_core_web_trf` (SpaCy)
- Jutsu Classification: `distilbert/distilbert-base-uncased`
- Character Chatbot: `meta-llama/Meta-Llama-3-8B-Instruct` (Base model)

## 📋 Requirements

- Python 3.12+
- CUDA-compatible GPU (recommended)
- Hugging Face API token (for model access)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/buriihenry/ninjaNLP.git
cd ninjaNLP
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required SpaCy model:
```bash
python -m spacy download en_core_web_trf
```

5. Set up environment variables:
Create a `.env` file in the root directory with:
```
HUGGINGFACE_TOKEN=your_token_here
```

## 🚀 Usage

### Running the Main Application

The main application provides a Gradio interface for all features:

```bash
python theme_app.py
```

This will launch a web interface with four main sections:

1. Theme Classification
2. Character Network Analysis
3. Text Classification (Jutsu)
4. Character Chatbot

### Individual Components

#### Theme Classification
```python
from theme_classifier import ThemeClassifier

classifier = ThemeClassifier(theme_list=["action", "friendship", "determination"])
themes = classifier.get_themes("path/to/subtitles", "output/path")
```

#### Character Network
```python
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator

ner = NamedEntityRecognizer()
network_gen = CharacterNetworkGenerator()
ners = ner.get_ners("path/to/subtitles", "ner/output/path")
network = network_gen.generate_character_network(ners)
```

#### Jutsu Classification
```python
from text_classification import JutsuClassifier

classifier = JutsuClassifier(
    model_path="your/model/path",
    data_path="path/to/jutsu/data",
    huggingface_token="your_token"
)
result = classifier.classify_jutsu("Description of jutsu")
```

#### Character Chatbot
```python
from character_chatbot import CharacterChatbot

chatbot = CharacterChatbot(
    model_path="burii/Naruto_Llama-3-8B",
    huggingface_token="your_token"
)
response = chatbot.chat("Your message", history=[])
```

## 📁 Project Structure

```
ninjaNLP/
├── theme_app.py              # Main Gradio application
├── theme_classifier/         # Theme classification module
├── character_network/        # Character network analysis
├── text_classification/      # Jutsu classification
├── character_chatbot/        # Naruto chatbot implementation
└── data/                     # Dataset directory
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for providing the transformer models
- SpaCy for Named Entity Recognition
- Gradio for the web interface
