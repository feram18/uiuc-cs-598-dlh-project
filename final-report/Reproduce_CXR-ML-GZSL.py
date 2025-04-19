import os
import numpy as np
import torch
from ChexnetTrainer import ChexnetTrainer
import argparse 

def main():
    args = argparse.Namespace(
        pretrained=True,
        vision_backbone='densenet121',
        save_dir='checkpoints',
        epochs=40,
        lr=0.0001,
        beta_rank=1,
        beta_map=0.01,
        beta_con=0.01,
        bce_only=True,
        resume_from=None,
        load_from=None,
        crop=224,
        train_file='dataset_splits/train.txt',
        num_classes=14,
        neg_penalty=0.20,
        textual_embeddings='embeddings/nih_chest_xray_biobert.npy',
        data_root='C:/Users/Mario/Classes/DL4H/Project/final-report/CXR8/CXR8/images'
    )

    seed = 1002
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        os.mkdir(args.save_dir)
    except OSError as error:
        print(error)

    trainer = ChexnetTrainer(args)
    trainer()

    # Load checkpoint from min loss model and test
    checkpoint = torch.load(f'{args.save_dir}/min_loss_checkpoint.pth.tar')
    trainer.model.load_state_dict(checkpoint['state_dict'])
    print('Testing the min loss model')
    test_ind_auroc = trainer.test()
    test_ind_auroc = np.array(test_ind_auroc)

    trainer.print_auroc(
        test_ind_auroc[trainer.test_dl.dataset.seen_class_ids],
        trainer.test_dl.dataset.seen_class_ids,
        prefix='\ntest_seen'
    )
    trainer.print_auroc(
        test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids],
        trainer.test_dl.dataset.unseen_class_ids,
        prefix='\ntest_unseen'
    )

    # Load checkpoint from best AUROC model and test
    checkpoint = torch.load(f'{args.save_dir}/best_auroc_checkpoint.pth.tar')
    trainer.model.load_state_dict(checkpoint['state_dict'])
    print('Testing the best AUROC model')
    test_ind_auroc = trainer.test()
    test_ind_auroc = np.array(test_ind_auroc)

    trainer.print_auroc(
        test_ind_auroc[trainer.test_dl.dataset.seen_class_ids],
        trainer.test_dl.dataset.seen_class_ids,
        prefix='\ntest_seen'
    )
    trainer.print_auroc(
        test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids],
        trainer.test_dl.dataset.unseen_class_ids,
        prefix='\ntest_unseen'
    )

if __name__ == '__main__':
    main()



