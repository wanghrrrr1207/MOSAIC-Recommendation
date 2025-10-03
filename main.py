import argparse
import os
import json
import re
import random
import numpy as np
import torch
from collections import defaultdict
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging
import warnings

# from mmoe_tr_eval import MMOE_trainer
from train_eval import group_MMOE_trainer
from models.mmoe_attn import MultiTaskModel
from data.data_prepare import load_group_data, load_group_data_save_index


# Initialize logging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
log_filename = f"training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Optional: Print logs to console as well
logging.getLogger().addHandler(logging.StreamHandler())





def parse_args():
    parser = argparse.ArgumentParser(description="Multitask Model for Rating and Review Generation")
    base_path = "/home/Your-Base-Path"
    parser.add_argument('--base_path', type=str, default=base_path, help="Base path to all data files")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size")
    parser.add_argument('--emb_size', type=int, default=64, help="Embedding size")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--num_experts', type=int, default=3, help="Number of experts")
    parser.add_argument('--num_attributes', type=int, default=6, help="Number of attributes")
    #Yelp/sports:7 beuty/tripad:6
    parser.add_argument('--sentiment_loss', type=int, default=1, help="set value more than 0 to use this loss")
    parser.add_argument('--use_cuda', action='store_true', help="Use CUDA if available")
    parser.add_argument("--model_name", type=str, default="MMOE",
                        help="Choose a model: MMOE, BaselineA, BaselineB")
    parser.add_argument('--num_group_users', type=int, default=15, help="Number of group users in ablation study")
    parser.add_argument('--alpha', type=float, default=0.0, help="opinion_loss parameter")
    parser.add_argument('--belta', type=int, default=30, help="cosine_sum parameter")
    parser.add_argument("--save_dir", type=str, default="./yelp_ab/MMOE_trip",
                        help="Directory to save best model checkpoints")

    # parser.add_argument('--gpu', type=int, default=1, help="GPU id to use")
    args = parser.parse_args()

    return args




def main():
    warnings.filterwarnings("ignore")
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    print("Loading rating and opinions matrix...")
    group_data = torch.load("/home/Your-Data-Path")

    # Load item2idx mapping
    #Yelp
    # print("Load Yelp user_item idx...")
    # with open("./new_processed_data/item2idx.json", "r") as f:
    #     item2idx = json.load(f)
    # with open("./new_processed_data/user2idx.json", "r") as f:
    #     user2idx = json.load(f)
    #beauty
    print("Load Amazon beauty user_item idx...")
    with open("./data/Amazon_mmoe/beauty_item2idx.json", "r") as f:
        item2idx = json.load(f)
    with open("./data/Amazon_mmoe/beauty_user2idx.json", "r") as f:
        user2idx = json.load(f)

    # print("Load Tripadvisor user_item idx...")
    # with open("./data/trip_mmoe/trip_item2idx.json", "r") as f:
    #     item2idx = json.load(f)
    # with open("./data/trip_mmoe/trip_user2idx.json", "r") as f:
    #     user2idx = json.load(f)
    # sports
    # print("Load Amazon sports user_item idx...")
    # with open("./data/Amazon_mmoe/sports_item2idx.json", "r") as f:
    #     item2idx = json.load(f)
    # with open("./data/Amazon_mmoe/sports_user2idx.json", "r") as f:
    #     user2idx = json.load(f)

#======================================================================

    # Prepare DataLoaders
    print("Prepare Dataloaders...")

    train_loader, val_loader, test_loader = load_group_data_save_index(group_data, batch_size=args.batch_size)

    num_users = len(user2idx.values()) 
    num_items = len(item2idx.values())


    if args.model_name == "MMOE":
        print("Initializing MMOE model...")
        mmoe_model = MultiTaskModel(
            num_users,
            num_items,
            input_dim=512, #448
            # emb_dim=args.emb_size,
            expert_dim=64,
            num_experts=args.num_experts,
            sentiment_loss=args.sentiment_loss,
            num_attributes=args.num_attributes,
            alpha = args.alpha,
            belta = args.belta,

            # max_label_size=max_label_size
        ).to(device)

    # Train Model
    mmoe_optimizer = torch.optim.AdamW(mmoe_model.parameters(), lr=0.00001, weight_decay=1e-4)
    # mmoe_optimizer = torch.optim.Adam(mmoe_model.parameters(), lr=0.0003)
    # mmmoe_scheduler = torch.optim.lr_scheduler.StepLR(mmoe_optimizer, step_size=5, gamma=0.25)
    scheduler = torch.optim.lr_scheduler.StepLR(mmoe_optimizer, 10, gamma=0.25) #yelp:5

    #=======group data training
    trainer = group_MMOE_trainer(mmoe_model, train_loader, val_loader, test_loader, mmoe_optimizer, scheduler, device, args.num_group_users,args.save_dir)

    # === Train & Evaluate Model ===
    num_epochs = 50
    trainer.train(num_epochs, is_predicted_opinion = True)
    #
    # === Final Testing ===
    trainer.test(is_predicted_opinion=True)

if __name__ == "__main__":
    main()
