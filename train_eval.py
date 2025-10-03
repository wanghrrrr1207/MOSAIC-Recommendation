import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
from recbole.config import Config
from recbole.utils import EvaluatorType, calculate_valid_score

from collections import defaultdict
# import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from metrics import root_mean_square_error, mean_absolute_error
# import seaborn as sns
import numpy as np
import random

class group_MMOE_trainer:

    """
    Trainer class for running experiments with ExpGCN and other baselines.
    Supports rating prediction (MSELoss) and attribute-opinion ranking (CrossEntropyLoss).
    Uses RecBole Evaluator for Recall@K, NDCG@K, and MSE.
    """

    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, device, num_group_users, save_dir="./saved_models"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.num_group_users = num_group_users

        # Initialize best model tracking
        self.best_valid_score = float("inf")  # Track highest Recall@10
        self.best_valid_result = None
        self.best_model_path = os.path.join(self.save_dir, f"best_{self.model.__class__.__name__}.pth")

        os.makedirs(self.save_dir, exist_ok=True)  # Ensure save directory exists

    def preprocess_labels(self, one_hot_labels):

        is_valid = (one_hot_labels.sum(dim=-1)) > 0  # [batch_size, num_attributes]

                 
        label_indices = torch.full_like(is_valid, -1, dtype=torch.long)  # [batch_size, num_attributes]

        if is_valid.any():
            valid_labels = one_hot_labels[is_valid]  # [num_valid, 4]
        label_indices[is_valid] = torch.argmax(valid_labels, dim=1)

        return label_indices

    import torch
    import random

    def ablation_aggregate_group_labels(self, group_labels):
        """
        Args:
            group_labels: list of B tensors, each [G, 7, 4]
        Returns:
            aggregated: Tensor [B, 7, 4] one-hot (or all-zero if no label)
            label_sum_all: Tensor [B, 7, 4] summed group labels (after truncating to max G=5)
        """
        aggregated = []
        label_sum_all = []

        for g_labels in group_labels:  # each is [G, 7, 4]
            g_labels = torch.stack(g_labels)
            G = g_labels.shape[0]

            # If G > 5, randomly select 5 groups
            if G > self.num_group_users:
                indices = random.sample(range(G), self.num_group_users)
                g_labels = g_labels[indices]

            label_sum = g_labels.sum(dim=0)  # [7, 4]
            label_sum_all.append(label_sum.unsqueeze(0))  # accumulate sum for this sample

            num_attributes = label_sum.shape[0]
            attr_valid_mask = (label_sum.sum(dim=-1) > 0)  # [7]

            one_hot = torch.zeros_like(label_sum)  # [7, 4]

            if attr_valid_mask.any():
                max_indices = label_sum.argmax(dim=-1)  # [7]
                for i in range(num_attributes):
                    if attr_valid_mask[i]:
                        one_hot[i, max_indices[i]] = 1.0

            aggregated.append(one_hot.unsqueeze(0))  # [1, 7, 4]

        return torch.cat(aggregated, dim=0), torch.cat(label_sum_all, dim=0)

    def aggregate_group_labels(self, group_labels):
        """
        Args:
            group_labels: list of B tensors, each [G, 7, 4]
        Returns:
            aggregated: Tensor [B, 7, 4] one-hot (or all-zero if no label)
        """
        aggregated = []

        for g_labels in group_labels:  # [G, 7, 4]
            # label_sum = g_labels.sum(dim=0)  # [7, 4]

            label_sum = torch.stack(g_labels).sum(dim=0)  # [7, 4]
            num_attributes = label_sum.shape[0]
            attr_valid_mask = (label_sum.sum(dim=-1) > 0)  # [7] 是否有label

            # 生成全零标签
            one_hot = torch.zeros_like(label_sum)  # [7, 4]

            # 只对有效属性进行投票
            if attr_valid_mask.any():
                max_indices = label_sum.argmax(dim=-1)  # [7]
                for i in range(num_attributes):
                    if attr_valid_mask[i]:
                        one_hot[i, max_indices[i]] = 1.0  # 只赋值有效位置

            aggregated.append(one_hot.unsqueeze(0))  # [1, 7, 4]

        return torch.cat(aggregated, dim=0)  # [B, 7, 4]


    def train(self, num_epochs, is_predicted_opinion=False):
        """
        Train the model for a given number of epochs and save the best model based on Recall@10.
        """
        self.model.train()
        trigger_times = 0

        for epoch in range(num_epochs):
            total_loss = []

            count = 0
            # self.peter_train()

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training"):
                user_ids = batch['user_idx'].to(self.device)  # [B]
                item_ids = batch['item_idx'].to(self.device)  # [B]
                ratings = batch['rating'].to(self.device)  # [B]
                opinion_labels = batch['user_opinion_label'].to(self.device)  # [B, 7, 4]
                # single_label_indices = self.preprocess_labels(opinion_labels)  # [B, 7] (optional)

                group_ids = batch['group_user_ids']  # list of B lists
                # group_ids = torch.stack([torch.tensor(g, device=item_ids.device) for g in group_ids], dim=0)  # [B, G]

                group_labels = batch['group_labels']
                gt_group_labels = self.aggregate_group_labels(group_labels).to(self.device)  # [B, 7, 4]

                group_sum_labels = batch['group_sum_label'].to(self.device)  # [B, 7, 4]
                group_label_indices = self.preprocess_labels(gt_group_labels)

                self.optimizer.step()


                loss = self.model.calculate_loss(user_ids, item_ids, group_ids, ratings, opinion_labels, group_sum_labels, gt_group_labels, group_label_indices, is_predicted_opinions=True)

                loss.backward()
                #===================
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss.append(loss.item())
                # total_loss.append(0)
                count+=1

            avg_loss = sum(total_loss) / len(self.train_loader)
            # self.scheduler.step(avg_loss)  #reduce lr
            self.scheduler.step()
            print('Learning rate set to {:2.8f}'.format(self.scheduler.get_last_lr()[0]))

            if is_predicted_opinion:
                org_rating_loss, val_opinion_loss, valid_results = self.evaluate(self.val_loader, is_predicted_opinion=True)
            else:
                org_rating_loss, valid_results = self.evaluate(self.val_loader, is_predicted_opinion=False)
            rating_review_loss = calculate_valid_score(valid_results, "Val Loss")
            # total_loss = calculate_valid_score(valid_results, "only_opinions")
            # valid_score = calculate_valid_score(valid_results, "Accuracy for group opinions labels")
            # acc_user = calculate_valid_score(valid_results, "Accuracy for user opinions labels")
            #


            print(f"Epoch {epoch+1}: Train Loss(rating+attr) = {avg_loss:.4f}, Val Rating Loss = {org_rating_loss:.4f}")
            print("🎯 Final Eval Results:", valid_results)
            # === Save Best Model Based on MSE loss ===
            if rating_review_loss < self.best_valid_score and rating_review_loss < org_rating_loss:
                self.best_valid_score = rating_review_loss

            # if org_rating_loss < self.best_valid_score:
            #     self.best_valid_score = org_rating_loss
                self.best_valid_result = valid_results
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"✅ Best model saved at {self.best_model_path} with best validation score of {self.best_valid_score}")

            else:
                trigger_times += 1
                if trigger_times >= 5:
                    print("Early stopping!")
                    print(trigger_times)
                    break
        # save_dir = "plots/mmoe_loss.png"
        # self.plot_loss_curves(plot_train_loss, plot_val_loss, save_path=save_dir)
    def evaluate(self, data_loader, is_predicted_opinion=False):
        """
        Evaluate the model using RecBole's Evaluator.
        Returns the validation loss and evaluation metrics.
        """
        self.model.eval()
        total_rating_loss = []
        ratings_opinions = []
        total_losses = []
        total_group_attr_acc = 0
        total_user_attr_acc = 0


        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                user_ids = batch['user_idx'].to(self.device)  # [B]
                item_ids = batch['item_idx'].to(self.device)  # [B]
                ratings = batch['rating'].to(self.device)  # [B]
                opinion_labels = batch['user_opinion_label'].to(self.device)  # [B, 7, 4]
                # batch_label_indices = self.preprocess_labels(opinion_labels)  # [B, 7] (optional)

                group_ids = batch['group_user_ids']  # list of B lists
                group_sum_labels = batch['group_sum_label'].to(self.device)
                group_labels = batch['group_labels']

                gt_group_labels = self.aggregate_group_labels(group_labels).to(self.device)  # [B, 7, 4]
                batch_label_indices = self.preprocess_labels(gt_group_labels)

                predicted_ratings = self.model.predict(user_ids, item_ids) #[batch_size, max_num_items]


                rating_loss = F.mse_loss(predicted_ratings, ratings)
                total_rating_loss.append(rating_loss.item())


 #=============================attributes evaluation =================================
                if is_predicted_opinion:
                    rating_review_loss, total_loss, group_predicted_labels, user_predicted_labels = self.model.predict_opinions(user_ids, item_ids, group_ids, ratings, opinion_labels, group_sum_labels, gt_group_labels, batch_label_indices)

                # true_opinions = opinion_data[:, :, 2:, :]

                    ratings_opinions.append(rating_review_loss.item())
                    total_losses.append(total_loss.item())

                    #we calculate accuracy for user-item opinion labels, not group labels
                    # group_attr_accuracy = self.compute_accuracy(gt_group_labels, group_predicted_labels)
                    group_attr_accuracy = self.compute_accuracy(opinion_labels, group_predicted_labels)
                    user_attr_accuracy = self.compute_accuracy(opinion_labels, user_predicted_labels)
                    total_group_attr_acc += group_attr_accuracy
                    total_user_attr_acc += user_attr_accuracy

                else:
                    total_group_attr_acc = 0
                    total_user_attr_acc = 0
                    ratings_opinions = 0


            # Compute accuracy for attribute prediction by checking position match
            avg_opinion_loss = sum(ratings_opinions) / len(data_loader)  #each epoch
            avg_total_loss = sum(total_losses) / len(data_loader)
            avg_rating_loss = sum(total_rating_loss) / len(data_loader)
            avg_user_label_acc = total_user_attr_acc / len(data_loader)
            avg_group_label_acc = total_group_attr_acc / len(data_loader)

        eval_results = {
            "original rating loss": avg_rating_loss,
            "Val Loss": avg_opinion_loss,
            "only_opinions": avg_total_loss,
            "Accuracy for user opinions labels": avg_user_label_acc,
            "Accuracy for group opinions labels": avg_group_label_acc
        }
        if is_predicted_opinion:
            return avg_rating_loss, avg_opinion_loss, eval_results
        else:
            return avg_rating_loss, eval_results  #plot val loss compared to train loss


    def test(self, is_predicted_opinion=False):
        """
        Load the best model and perform final testing.
        """
        print(f"🔍 Loading best model from {self.best_model_path} for final testing...")
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()

        total_group_attr_acc = 0
        total_user_attr_acc = 0

        total_mae = 0
        total_rmse = 0
        total_mae_r = 0
        total_mae_g_r_w_r = 0
        total_mae_u_r_w_r = 0
        total_rmse_r=0
        total_rmse_g_r_w_r = 0
        total_rmse_u_r_w_r = 0

        all_gt = []
        all_pred = []
        group_pred_ratings = []
        true_ratings = []


        with torch.no_grad():

            # RMSE, MAE = self.peter_test()
            for batch in tqdm(self.test_loader, desc="Testing"):
                user_ids = batch['user_idx'].to(self.device)  # [B]
                item_ids = batch['item_idx'].to(self.device)  # [B]
                ratings = batch['rating'].to(self.device)  # [B]
                opinion_labels = batch['user_opinion_label'].to(self.device)  # [B, 7, 4]
                # batch_label_indices = self.preprocess_labels(opinion_labels)  # [B, 7] (optional)

                group_ids = batch['group_user_ids']  # list of B lists

                # group_ids = torch.stack([torch.tensor(g, device=item_ids.device) for g in group_ids], dim=0)  # [B, G]
                group_labels = batch['group_labels']
                group_sum_labels = batch['group_sum_label'].to(self.device)
                gt_group_labels = self.aggregate_group_labels(group_labels).to(self.device)  # [B, 7, 4]
                batch_label_indices = self.preprocess_labels(gt_group_labels)

                predicted_ratings = self.model.predict(user_ids, item_ids)  # [batch_size, max_num_items]

                # true_ratings.extend(ratings.tolist())
                pred_ratings = [(r, p) for (r, p) in zip(ratings.tolist(), predicted_ratings.tolist())]
                rmse = root_mean_square_error(pred_ratings, 5, 1)
                total_rmse += rmse
                mae = mean_absolute_error(pred_ratings, 5, 1)
                total_mae += mae

    #===================opinion test==========================
                if is_predicted_opinion:
                    predicted_ratings_org, group_pred_labels, user_pred_labels, user_rating_with_reviews, group_rating_with_reviews = self.model.predict_rating_with_reviews(user_ids, item_ids, group_ids, ratings, gt_group_labels, group_sum_labels, batch_label_indices)

                    pred_ratings_org = [(r, p) for (r, p) in zip(ratings.tolist(), predicted_ratings_org.tolist())]
                    g_r_w_r = [(r, p) for (r, p) in zip(ratings.tolist(), group_rating_with_reviews.tolist())]
                    u_r_w_r = [(r, p) for (r, p) in zip(ratings.tolist(), user_rating_with_reviews.tolist())]

                    group_pred_ratings.extend(group_rating_with_reviews.tolist())
                    true_ratings.extend(ratings.tolist())

                    rmse_r = root_mean_square_error(pred_ratings_org, 5, 1)
                    rmse_g_r_w_r = root_mean_square_error(g_r_w_r, 5, 1)
                    rmse_u_r_w_r = root_mean_square_error(u_r_w_r, 5, 1)
                    total_rmse_r += rmse_r
                    total_rmse_g_r_w_r += rmse_g_r_w_r
                    total_rmse_u_r_w_r += rmse_u_r_w_r
                    mae_r = mean_absolute_error(pred_ratings_org, 5, 1)
                    mae_g_r_w_r = mean_absolute_error(g_r_w_r, 5, 1)
                    mae_u_r_w_r = mean_absolute_error(u_r_w_r, 5, 1)

                    total_mae_r += mae_r
                    total_mae_g_r_w_r += mae_g_r_w_r
                    total_mae_u_r_w_r += mae_u_r_w_r

                    #===========user and group label accuracy===========

                    total_group_attr_acc += self.compute_accuracy(opinion_labels, group_pred_labels)
                    total_user_attr_acc += self.compute_accuracy(opinion_labels, user_pred_labels)
                    all_gt.append(opinion_labels)  # gt_label: tensor
                    all_pred.append(group_pred_labels)

                else:
                    total_group_attr_acc = 0
                    total_user_attr_acc = 0
                    total_rmse_r = 0
                    total_mae_r = 0
                    total_mae_r_w_r = 0
                    total_rmse_r_w_r = 0

            all_gt = torch.cat(all_gt, dim=0)  # shape: [Total_samples, 7, 4]
            all_pred = torch.cat(all_pred, dim=0)  # shape: [Total_samples, 7, 4]
            group_metrics = self.evaluate_opinion_metrics(all_gt, all_pred)

            rating_RMSE = total_rmse / len(self.test_loader)
            rating_MAE = total_mae / len(self.test_loader)
            avg_mae = total_mae / len(self.test_loader)
            # new_rating_RMSE = total_rmse_r / len(self.test_loader)
            # new_rating_MAE = total_mae_r / len(self.test_loader)
            rating_with_group_reviews_RMSE = total_rmse_g_r_w_r / len(self.test_loader)
            rating_with_user_RMSE = total_rmse_u_r_w_r / len(self.test_loader)
            rating_with_group_reviews_MAE = total_mae_g_r_w_r / len(self.test_loader)
            rating_with_user_MAE = total_mae_u_r_w_r / len(self.test_loader)
            user_label_acc = total_user_attr_acc / len(self.test_loader)
            group_label_acc = total_group_attr_acc / len(self.test_loader)

        # eval_results = collector.get_result()alb

        test_results = {

            "only test rating RMSE:": rating_RMSE,
            "only test rating MAE:": rating_MAE,
            # "new rating RMSE:": new_rating_RMSE,
            # "new rating MAE:": new_rating_MAE,
            "rating_with_user_RMSE:": rating_with_user_RMSE,
            "rating_with_group_reviews_RMSE:": rating_with_group_reviews_RMSE,
            "rating_with_user_MAE:": rating_with_user_MAE,
            "rating_with_group_reviews_MAE:": rating_with_group_reviews_MAE,
            "Accuracy for user opinions labels": user_label_acc,
            "Accuracy for group opinions labels": group_label_acc
        }
        print("🎯 Final Test Results:", test_results)
        print("Lable test metrics", group_metrics)

        return rating_RMSE, test_results


    # def plot_loss_curves(self, train_losses, val_losses, save_path):
    #     """
    #     Plots and saves training and validation loss curves on the same figure.
    #
    #     Args:
    #         train_losses (list): Training loss per epoch.
    #         val_losses (list): Validation loss per epoch.
    #         save_path (str): Path to save the plot.
    #     """
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     epochs = range(1, len(train_losses) + 1)
    #
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
    #     plt.plot(epochs, val_losses, label='Validation Loss', marker='s', color='orange')
    #
    #     plt.title("Training & Validation Loss Over Epochs")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(save_path)
    #     plt.close()
    #
    #     print(f"📉 Loss curves saved to {save_path}")



    def evaluate_opinion_metrics(self, opinion_labels, group_pred_labels):
        """
        opinion_labels: torch.Tensor [B,7,4], GT one-hot
        group_pred_labels: torch.Tensor [B,7,4], predicted probabilities
        Returns: dict of precision, recall, f1, auc per attribute and overall
        """
        device = opinion_labels.device
        B, A, C = opinion_labels.shape

        # ===== Convert to numpy =====
        probs = group_pred_labels.detach().cpu().numpy()  # [B,7,4]

        # ===== Convert GT to indices =====
        gt_indices = opinion_labels.argmax(dim=-1).view(-1).cpu().numpy()  # [B*7]

        # ===== Predicted indices =====
        pred_indices = group_pred_labels.argmax(dim=-1).view(-1).cpu().numpy()  # [B*7]

        # ===== Mask invalid labels =====
        valid_mask = (opinion_labels.sum(dim=-1) > 0).view(-1).cpu().numpy()  # [B*7]

        metrics = {}
        all_prec, all_rec, all_f1, all_auc = [], [], [], []

        # ===== Overall (macro) metrics =====
        if valid_mask.sum() > 0:
            overall_prec = precision_score(gt_indices[valid_mask], pred_indices[valid_mask], average='macro',
                                           zero_division=0)
            overall_rec = recall_score(gt_indices[valid_mask], pred_indices[valid_mask], average='macro',
                                       zero_division=0)
            overall_f1 = f1_score(gt_indices[valid_mask], pred_indices[valid_mask], average='macro', zero_division=0)

        #
        else:
            overall_prec = overall_rec = overall_f1 = 0.0

        metrics['overall_precision'] = overall_prec
        metrics['overall_recall'] = overall_rec
        metrics['overall_f1'] = overall_f1
        # metrics['overall_auc'] = overall_auc

        # ===== Per attribute metrics =====
        # for attr in range(A):
        #     gt_attr = opinion_labels.argmax(dim=-1)[:, attr].cpu().numpy()  # [B]
        #     pred_attr = group_pred_labels.argmax(dim=-1)[:, attr].cpu().numpy()  # [B]
        #     valid_attr = (opinion_labels.sum(dim=-1)[:, attr] > 0).cpu().numpy()  # [B]
        #
        #     if valid_attr.sum() > 0:
        #         p = precision_score(gt_attr[valid_attr], pred_attr[valid_attr], average='macro', zero_division=0)
        #         r = recall_score(gt_attr[valid_attr], pred_attr[valid_attr], average='macro', zero_division=0)
        #         f = f1_score(gt_attr[valid_attr], pred_attr[valid_attr], average='macro', zero_division=0)
        #
        #     else:
        #         p = r = f = auc = 0.0
        #
        #     metrics[f'attr{attr}_precision'] = p
        #     metrics[f'attr{attr}_recall'] = r
        #     metrics[f'attr{attr}_f1'] = f
        #     # metrics[f'attr{attr}_auc'] = auc
        #
        #     all_prec.append(p)
        #     all_rec.append(r)
        #     all_f1.append(f)
        #     # all_auc.append(auc)

        return metrics

    def compute_accuracy(self, true_labels, predicted_labels):
        """
        true_labels: [B, 7, 4] one-hot
        predicted_labels: [B, 7, 4] logits or probs
        """
        with torch.no_grad():
            valid_mask = true_labels.sum(dim=-1) > 0  # [B, 7]

            # 获取预测类别：取最大概率的位置
            pred_classes = torch.argmax(predicted_labels, dim=-1)  # [B, 7]
            true_classes = torch.argmax(true_labels, dim=-1)  # [B, 7]

            # 只保留有效标签位置
            correct = (pred_classes == true_classes) & valid_mask  # [B, 7]
            num_correct = correct.sum().item()
            num_valid = valid_mask.sum().item()

            accuracy = num_correct / num_valid if num_valid > 0 else 0.0
            return accuracy


