import torch
import huggingface_hub
import pandas as pd
from . cleaner import Cleaner
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments,
                          pipeline)
from datasets import Dataset
from . training_utils import get_class_weights, compute_metrics
import gc
from . custom_trainer import CustomTrainer

class JutsuClassifier():

    def __init__(self,
                model_path,
                data_path=None,
                text_column_name="text",
                label_column_name="justus",
                model_name="distilbert/distilbert-base-uncased",
                test_size=0.2,
                num_labels=3,
                huggingface_token=None,
                ):
        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_labels = num_labels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.huggingface_token = huggingface_token
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token) 

        self.tokenizer = self.load_tokenizer()
        
        # Check if model_path is a valid Hugging Face repo ID
        is_hf_repo = False
        if not self.model_path.startswith('/'):  # Not a local absolute path
            try:
                is_hf_repo = huggingface_hub.repo_exists(self.model_path)
            except Exception as e:
                is_hf_repo = False
        
        # If it's a valid HF repo or if we need to train (local data path provided)
        if is_hf_repo or (not is_hf_repo and self.data_path is not None):
            if self.data_path is None:
                raise ValueError("Data path is required to train a model, since the model is not available on huggingface hub.")
            train_data, test_data = self.load_data(self.data_path)
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()
            all_data = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(all_data)

            self.train_model(train_data, test_data, class_weights)
        
        self.model = self.load_model(self.model_path)
        

    def load_model(self, model_path):
        model = pipeline(
            "text-classification", 
            model=model_path, 
            return_all_scores=True,
            )
        return model

    def train_model(self, train_data, test_data, class_weights):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, 
                                                                   num_labels=self.num_labels,
                                                                   id2label=self.label_dict,
                                                                   )    
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            push_to_hub=True,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        
        trainer.set_device(self.device)
        trainer.set_class_weights(class_weights)
        trainer.train() 

        # Flush Memory
        del trainer,model
        gc.collect()

        if self.device == 'cuda':
            torch.cuda.empty_cache()



 

    def simplified_jutsu_type(self,jutsu):
        if "Genjutsu" in jutsu:
            return "Genjutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
        if "Taijutsu" in jutsu:
            return "Taijutsu"
    def preprocess_function(self,tokenizer, examples):
        return tokenizer(examples["text_cleaned"], truncation=True)    
                

    def load_data(self,data_path):
            df = pd.read_json(data_path, lines=True)
            df['justu_type_simplified'] = df['jutsu_type'].apply(self.simplified_jutsu_type)
            df['text'] = df['jutsu_name'] + ". " + df['jutsu_description']
            df[self.label_column_name] = df['justu_type_simplified']
            df = df[['text',self.label_column_name]]
            df = df.dropna()
         

            # clean text
            cleaner = Cleaner()
            df['text_cleaned'] = df[self.text_column_name].apply(cleaner.clean)

            # Encode labels
            le = preprocessing.LabelEncoder()
            le.fit(df[self.label_column_name].tolist())

            label_dict = {index: label_name for index, label_name in enumerate(le.__dict__['classes_'].tolist())}
            self.label_dict = label_dict
            df['label'] = le.transform(df[self.label_column_name].tolist())


            #train test split
            test_size = 0.2
            df_train, df_test = train_test_split(df,
                                                test_size=test_size, 
                                                stratify=df['label'],
                                                )
                        # convert pandas to huggingface dataset
            train_dataset = Dataset.from_pandas(df_train)
            test_dataset = Dataset.from_pandas(df_test)

            #tokenize dataset
            tokenized_train = train_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples), batched=True)
            tokenized_test = test_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples), batched=True)

            return tokenized_train, tokenized_test



    def load_tokenizer(self):
    # Check if model_path is a Hugging Face Hub repository
        if self.model_path.startswith('http') or ('/' in self.model_path and not self.model_path.startswith('/')):
            # It's potentially a HF repo
            try:
                if huggingface_hub.repo_exists(self.model_path):
                    tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception as e:
                # If there's an error validating the repo, fall back to default model
                print(f"Error validating repository, using default model: {e}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            # It's likely a local path, use the default model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def postprocess(self, model_output):
        output = []
        for pred in model_output:
            label = max(pred,key=lambda x:x['score'])['label']
            output.append(label)
        return output
    
    def classify_jutsu(self, text):
        model_output = self.model(text)
        predictions = self.postprocess(model_output)
        return predictions