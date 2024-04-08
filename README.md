# Assignment2_PARTB
using pre-trained model (resnet50)
dl_assn2_partb consist of two files
1. dl_assn2_parb.ipynb
2. dl_assn2_partb.py

run dl_assn2_partb.py with the optinal arguments by using the wandb API key.
arguments are as follows

    '-wp', '--wandb_project', type=str, default="Assignment2_PartB", help="wandb project name"
    
    '-opt', '--optim_name', type=str, default="nadam", choices = ['sgd','adam','nadam'], help="optimizer for backprop"
    
    '-bS', '--batchSize', type=int, default=32, choices = [32, 64], help="batch size")
    
    '-ag', '--data_aug', type=str, default="no", choices = ['yes', 'no'], help="data augmentation"
    
    '-nE', '--num_epochs', type=int, default=5, choices = [5, 10], help="number of epochs"
    
    '-lR', '--learning_rate', type=float, default=1e-3, choices = [1e-3, 1e-4], help="learning rate"
    
    
This initiates a call to train_cnn_ud() which will import the RESNET50 base model and fine-tune as per the requirement, this vill produce validation and test accuracy and logs it in wandb CS23M005/Assignment2_PARTB.

Detailed description of the functions:
1. parse_args()
     This function takes the input from the command line arguments and call train_cnn_ud() with required parameters.
     This function has default values in which case if user does not provide arguments
2. train_cnn_ud()
     This is the major function which will get the data loader from get_data()
     get the base model with modified for training from resnet50_ud()
     trains the model for each epoch by doing forward and backward propagations
     After training the model, it will calculate train accuracy/loss, validation accuracy/loss using check_accuracy()
3. resnet50_ud()
     This function will import the pretrained model and replaces the last layer with size of 10
     For fine tuning purposes, the last fully connected layer is made to learn from back propagation
4. get_data()
     This function will generate the train_loader/validation_loader from the iNaturalist dataset with the required   transformation (with and withou data augmentation)
5. check_accuracy()
     This function will do the plain forward propagation and calculate accuracy/loss w.r.t predictions vs targets
