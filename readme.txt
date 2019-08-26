1. To generate hashtags using soft attention model (Show, Attend Tell)

-- Harrison dataset is used which is preprocessed and split into (80:10:10) train/validation/test ration by preprocess.py file
-- Soft-attention model is trained using tensorflow_attention.py in the Show, attend and Tell (Soft Attention) directory.
-- The code in the tensorflow_attention.py is adapted from "https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/image_captioning.ipynb"
-- The test data results, loss_epoch, perplexity_epoch readings obtained from the after training the model are saved in the directory Show, attend and Tell (Soft Attention) directory.
-- The model requires Keras, Tensorflow and Python 3.6 to train. The requirements can be installed in anaconda environment using environment_tensorflow.yaml


2.  To generate hashtags using soft attention model (Show, Attend Tell)

-- Harrison dataset is used which is preprocessed and split into (80:10:10) train/validation/test ration by preprocess.py file
-- CNN-LSTM model ()defined in model.py) is used to train the model using train.py in the Show and tell directory.
-- Run build_vocab.py --> train.py to train the model. Run sample.py to generate test data results.
-- The code is adapted from "https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning"
-- Requires PyTorch setup which can be done by creating a pytorch environment using environment_pytorch.yaml

3. To generate hashtags using Multi-label image classification

-- train2.py is used to generate hashtags using AlexNet implemented using Keras by running in tensorflow environment.
-- The test data results, loss_epoch (train and validation), accuracy_epoch(train and validation) readings obtained after training and validation of the model are saved in the Multi-label image classification directory.

4. Character-level language model using RNN

-- model trained using tensorflow
-- model trained on PersonaBank corpus saved in new_persona.txt. (Preprocessed version of persona data) 
-- train model by running char_lang_model.py in tensorflow environment.
