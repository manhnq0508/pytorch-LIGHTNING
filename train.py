import getopt
from re import L
import sys
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from src import ActionClassificationLSTM, HandDataModule, WINDOW_SIZE

# DATASET_PATH = "/content/DataRoot/"

def configuration_parser(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # training batch size
    parser.add_argument('--batch_size', type=int, default=512)
    # max training epochs = 400
    parser.add_argument('--epochs', type=int, default=400)
    # training initial learning rate
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    # number of classes = number of human actions in the data set= 6
    parser.add_argument('--num_class', type=int, default=6)
    return parser


def do_training_validation(argv):
    opts, args = getopt.getopt(argv, "hd:", ["data_root="])
    try:
        opts, args = getopt.getopt(argv, "hd:", ["data_root="])
    except getopt.GetoptError:
        print ('train.py -d <data_root>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('train.py -d <data_root>')
            sys.exit()
        elif opt in ("-d", "--data_root"):
            data_root = arg
    print ('data_root is "', data_root)

    pl.seed_everything(21)    
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = configuration_parser(parser)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    # print(args)
    # init model
    hidden_dim = 50
    model = ActionClassificationLSTM(WINDOW_SIZE, hidden_dim, learning_rate=args.learning_rate)
    data_module = HandDataModule(batch_size=args.batch_size) 
    data_module.setup() 
    data_module.train_dataloader()
    data_module.val_dataloader()
    print(data_module.X_train.shape)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='val_loss')
    lr_monitor = LearningRateMonitor(logging_interval='step')    
    # most basic trainer, uses good defaults
    trainer = pl.Trainer.from_argparse_args(args,
        # fast_dev_run=True,
        max_epochs=args.epochs, 
        deterministic=True, 
        gpus=1, 
        progress_bar_refresh_rate=1, 
        callbacks=[EarlyStopping(monitor='train_loss', patience=15), checkpoint_callback, lr_monitor])    
    trainer.fit(model, data_module)    
    return model

def get_latest_run_version_ckpt_epoch_no(lightning_logs_dir='lightning_logs', run_version=None):
    if run_version is None:
        run_version = 0
        for dir_name in os.listdir(lightning_logs_dir):
            if 'version' in dir_name:
                if int(dir_name.split('_')[1]) > run_version:
                    run_version = int(dir_name.split('_')[1])                
    checkpoints_dir = os.path.join(lightning_logs_dir, 'version_{}'.format(run_version), 'checkpoints')    
    files = os.listdir(checkpoints_dir)
    ckpt_filename = "Save_model"
    for file in files:
        print(file)
        if file.endswith('.ckpt'):
            ckpt_filename = file        
    if ckpt_filename is not None:
        ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)
    else:
        print('CKPT file is not present')    
    return ckpt_path


if __name__ == '__main__':
    do_training_validation(sys.argv[1:])
    ckpt_path = get_latest_run_version_ckpt_epoch_no()
    print('The latest model path: {}'.format(ckpt_path))