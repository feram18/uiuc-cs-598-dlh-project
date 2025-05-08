import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer
import argparse 


def main ():

    #args = parse_args()
    args = argparse.Namespace(
        pretrained=False,
        vision_backbone='densenet121',
        save_dir='checkpoints',
        epochs=40,
        lr=0.0001,
        beta_rank=1,
        beta_map=0.1,
        beta_con=0.1,
        bce_only=False,
        resume_from=None,
        load_from="checkpoints/best_auroc_checkpoint.pth.tar",
        crop=224,
        train_file='dataset_splits/train.txt',
        num_classes=14,
        neg_penalty=0.03,
        batch_size=4,
        resize=256,
        wo_con=False,
        wo_map=False,
        val_file='dataset_splits/val.txt',
        test_file='dataset_splits/test.txt',
        steps='20, 40, 60, 80',
        textual_embeddings='embeddings/nih_chest_xray_biobert.npy',
        data_root='/mnt/c/Users/Mario/Classes/DL4H/Project/final-report/CXR8/CXR8/images/'
    )
    try:  
        os.mkdir(args.save_dir)  
    except OSError as error:
        print(error) 
    
    trainer = ChexnetTrainer(args)
    print ('Testing the trained model')
    

    test_ind_auroc = trainer.test()
    test_ind_auroc = np.array(test_ind_auroc)
    


    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')
            

if __name__ == '__main__':
    main()





