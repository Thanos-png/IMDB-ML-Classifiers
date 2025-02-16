These are the project requirements:
	numpy>=1.21,<2.0
	scikit-learn==1.6.1
	torch==2.1.0
	matplotlib==3.7.1
	torchtext==0.16.0
Also ensure GPU is available with (python -c "import torch; print(torch.cuda.is_available())")
Note: Models haven't been tested with CPU

For the AdaBoost Classifier:
1) Open your CLI
2) Move to the src directory:
cd src/
3) Run the following command to train the model:
python train_adaboost.py
4) Run the following command to test the model:
python test_adaboost.py

For the Random Forest Classifier:
1) Open your CLI
2) Move to the src directory:
cd src/
3) Run the following command to train the model:
python train_randomforest.py
4) Run the following command to test the model:
python test_randomforest.py

For the Stacked Bidirectional RNN:
1) Open your CLI
2) Move to the src directory:
cd src/
3) Run the following command to train the model:
python train_rnnmodel.py
4) Run the following command to test the model:
python test_rnnmodel.py
