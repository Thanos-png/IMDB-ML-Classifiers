To run the project requires the following python libraries:
	numpy>=1.21,<2.0
	scikit-learn==1.6.1
	torch==2.1.0
	matplotlib==3.7.1
	torchtext==0.16.0
Also you have to ensure GPU is available (with command python -c "import torch; print(torch.cuda.is_available())")

FOR RANDOM FOREST CLASSIFIER:
1) Open cmd
2) Move to src
3)Run the following command to train the model:
python train_randomforest.py
4)Run the following command to test the model:
python test_randomforest.py