Description:
Neural networks are typically trained on large-scale labeled datasets. Since labeling data is a slow and expensive process, it is important to carefully select which documents to annotate.

This assignment focuses on creating a method to identify which documents should be selected for annotation for a document classification task, using the DocILE dataset (available on GitHub). Only the documents in the provided training set are annotated at the start. The objective is to choose additional documents from the provided pool of unlabeled documents for annotation, so as to maximize classification accuracy on the validation set. You should validate your approach for unsupervised/semi-supervised document selection on the validation set, compare with relevant baselines, and document your experimental findings.

Detailed instructions:
- Register and download the annotated-trainval dataset from the provided link. The dataset is a compressed archive containing training and validation documents with labels. For this experiment, use the validation set (val.json, rename to new_train.json) as your training set, and use the training set (train.json, rename to unlabeled.json) as the pool of unlabeled documents.
- Implement a neural network for document classification in PyTorch or TensorFlow, using a vision-language model (VLM) or large language model (LLM) with fewer than 3 billion parameters, or use a publicly available implementation.
- Train an initial model on the entire dataset as a baseline (this serves as a performance ceiling and should not be used for document selection or flagging).
- Train a second model using only the training set; this is the baseline to improve upon.
- Select 1500 documents from the unlabeled pool (without using their labels) for annotation.
- After retrieving the annotations for these documents, train a third model using the training set plus the newly annotated documents.
- Compare the results obtained from different selection methods. Document the selection strategies used, analyze the experimental results, and share your findings by emailing hiring-ailabs@abbyy.com.
- For any questions regarding the assignment, please use the same email address.