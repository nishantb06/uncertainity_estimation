Uncertainty estimation

So this entire experiment is aimed at finding the answer to the question: “From a given pool of unlabeled data, how would you select a subset to annotate so that they can help your model become better?”

In this experiment we are working with a set of 5680 documents, out of which 500 documents make the validation set and the remaining 5180 make the training set. We are asked to train a model on the validation set first and then use this model to select 1500 documents from the training set, which is assumed to be unlabeled.

After we have chosen 1500 documents, we then use these 1500 together with the 500 from the validation set to train our model and prove that using this additional set of documents our model does perform better than the model only trained on the validation set.

Meanwhile, we also have to train a model using the entire set of 5680 documents to set a ceiling for our training runs. Our model trained with 1500 + 500 cannot outperform the ceiling model, but the model trained on just the validation set most probably won’t. This above statement is verified from our experiments as well.

Experiments setup

Infrastructure

I train on AWS spot instances (G6) machines and I keep one EBS volume which gets reattached and mounted to every new machine. This way all my checkpoints and logs persist across all the training sessions.

Model choice

Every recent SOTA document understanding and OCR model has a two-phase pipeline. The vision encoder usually consists of ~500M parameters in total and is used to encode visual information into token embeddings which are then consumed by the LLM decoder. It’s important to keep the visual embeddings count to a minimum because dividing an image into 16×16 patches can easily explode the context length.

Due to this reason I decided to go with DeepSeek’s vision encoder pipeline, which consists of an 80M SAM encoder followed by a convolution layer to downscale, followed by a 300M CLIP model which outputs either 64, 100, or 256 tokens depending on the input image configuration, which can be 512×512, 640×640, or 1024×1024.

I then decided to attach my own custom MLP on the CLS token which CLIP outputs for the classification task of this document data.

Experiments setup

Infrastructure

I train on AWS spot instances (G6) machines and I keep one EBS volume which gets reattached and mounted to every new machine. This way all my checkpoints and logs persist across all the training sessions.

Model choice

Every recent SOTA document understanding and OCR model has a two-phase pipeline. The vision encoder usually consists of ~500M parameters in total and is used to encode visual information into token embeddings which are then consumed by the LLM decoder. It’s important to keep the visual embeddings count to a minimum because dividing an image into 16×16 patches can easily explode the context length.

Due to this reason I decided to go with DeepSeek’s vision encoder pipeline, which consists of an 80M SAM encoder followed by a convolution layer to downscale, followed by a 300M CLIP model which outputs either 64, 100, or 256 tokens depending on the input image configuration, which can be 512×512, 640×640, or 1024×1024.

I then decided to attach my own custom MLP on the CLS token which CLIP outputs for the classification task of this document data.

All the documents are divided into 7 categories like tax invoice and orders and such. The two I just mentioned dominate the class distribution heavily, and so I have only included results from those two.

Given more time and resources, I could have done the same exercise on object detection loss function as well.

Methods for uncertainty estimation:

I primarily decided to go with the simplest approach I could think of.

1. Entropy estimation: If the entropy of the output of the MLP is too high, then that means that the model is unable to classify that document clearly. This means that if this document was in the training set, it would definitely help the model learn, i.e. provide informative gradients.

The above principle can be applied in a variety of combinations. For example:

1. Pass the same document through multiple resolutions. High entropy on all three, or extremely high entropy on one of those resolutions, probably indicates this document should be included in the training set.
2. Take multiple checkpoints of the same model, collect entropy information from all of them, and then compare.

3. Compare Jensen–Shannon divergence, which is symmetric in nature, across the logits of the same documents inferred at different resolutions.

Using the average entropy method, here are the results. It’s clear that the model trained on the 1500 + 500 set outperforms the model trained on just the validation set, but is not able to beat the ceiling set by training on the entire set.