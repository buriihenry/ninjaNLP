from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def get_class_weights(df):
   # Get unique labels
   unique_labels = df['label'].unique()
   
   # If unique_labels is not already a numpy array, convert it
   if not isinstance(unique_labels, np.ndarray):
       unique_labels = np.array(unique_labels)
   
   # Sort the labels (keeping as numpy array)
   sorted_unique_labels = np.sort(unique_labels)
   
   # Convert labels to numpy array if not already
   y_labels = np.array(df['label'].tolist()) if not isinstance(df['label'], np.ndarray) else df['label']
   
   class_weights = compute_class_weight(
       'balanced', 
       classes=sorted_unique_labels,  # This is now definitely a numpy array
       y=y_labels
   )
   return class_weights