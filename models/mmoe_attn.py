import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional
from torch import Tensor


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        output = src
        attns = []
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)
        attns = torch.stack(attns)
        if self.norm is not None:
            output = self.norm(output)
        return output, attns

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MLP_as_MMOE(nn.Module):
    def __init__(self, emsize=512, hidden_size=512, num_layers=2):
        super(MLP_as_MMOE, self).__init__()
        self.first_layer = nn.Linear(emsize, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, hidden):  # hidden: (batch_size, emsize)
        out = self.sigmoid(self.first_layer(hidden))  # -> (batch_size, hidden_size)
        for layer in self.layers:
            out = self.sigmoid(layer(out))            # -> (batch_size, hidden_size)
        rating = torch.squeeze(self.last_layer(out))  # -> (batch_size,)
        return rating


class MMOE_MLP(nn.Module):
    def __init__(self, emsize=512):
        super(MMOE_MLP, self).__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):
        mlp_vector = self.sigmoid(self.linear1(hidden))
        # rating = self.sigmoid(self.linear2(mlp_vector))
        rating = torch.squeeze(self.linear2(mlp_vector))
        # rating = rating.squeeze(-1)
        return rating


class MMOE_predictor(nn.Module):
    def __init__(self, emsize=512):
        super(MMOE_predictor, self).__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):
        mlp_vector = self.gelu(self.linear1(hidden))
        # rating = self.sigmoid(self.linear2(mlp_vector))
        score = self.linear2(mlp_vector).squeeze(-1)
        # score = torch.squeeze(self.linear2(mlp_vector))
        # rating = rating.squeeze(-1)
        return score

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, expert_dim, nhead=4, dropout_rate=0.1): #########0.1
        super(ExpertNetwork, self).__init__()
        # self.layer = nn.Linear(input_dim, expert_dim)
        self.layer = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            # nn.ReLU(),
            nn.GELU(),
            # nn.BatchNorm1d(expert_dim),
            nn.LayerNorm(expert_dim),
            nn.Dropout(dropout_rate)
        )
        nn.init.xavier_uniform_(self.layer[0].weight)

        # Transformer encoder layer
        self.transformer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=expert_dim*4,
            dropout=0.5    ####dropout=0.5
        )

        # Final MLP for expert output
        # self.mlp = MLP(emsize=expert_dim)

    def forward(self, x):
        self.layer(x)
        # Add sequence dimension for transformer (sequence length = 1)
        x = x.unsqueeze(0)  # [1, batch_size, expert_dim]

        # Process through transformer
        x, _ = self.transformer(x)  # [1, batch_size, expert_dim]

        # Remove sequence dimension
        x = x.squeeze(0)  # [batch_size, expert_dim]

        # Final MLP processing
        # x = self.mlp(x)  # [batch_size, 1]
        # x = x.squeeze(-1)  # [batch_size]

        return x

class GateNetwork(nn.Module):
    def __init__(self, num_experts, input_dim):
        super(GateNetwork, self).__init__()
        self.gate_layer = nn.Linear(input_dim, num_experts)
        # self.temperature = temperature
        nn.init.xavier_uniform_(self.gate_layer.weight)

    def forward(self, x):
        # Generate gating weights using softmax
        # return F.softmax(self.gate_layer(x), dim=1)

        return F.softmax(self.gate_layer(x), dim=1)

class ReviewAttentionNetwork(nn.Module):
    def __init__(self, input_dim, num_attributes, hidden_dim=64):
        super(ReviewAttentionNetwork, self).__init__()
        self.num_attributes = num_attributes
        self.query = nn.Linear(input_dim, input_dim)
        #============diff to be query============
        self.diff_query = nn.Linear(1, input_dim)
        self.diff_query_proj = nn.Sequential(
            nn.Linear(1, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim)
        )
        # Label-based queries (2 heads)
        num_labels = 4
        self.label_query_proj1 = nn.Linear(num_labels, input_dim)
        self.label_query_proj2 = nn.Linear(num_labels, input_dim)
        #============================================
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = math.sqrt(input_dim)


        initial_sentiment_map = torch.tensor([1.0, 0.7, -0.7, -1.0]).unsqueeze(1)  # [4,1]
        self.sentiment_embedding = nn.Parameter(initial_sentiment_map * torch.randn(4, input_dim))

        self.fuse_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
        self.review_score_predictor = MMOE_predictor(input_dim)


    def gt_review_score(self, true_labels, rating_feature, mask=None):
        probs = true_labels.float()
        batch_size = probs.size(0)
        sentiment_vecs = torch.matmul(probs, self.sentiment_embedding)  # [B, 7, D]

        if mask is not None:
            sentiment_vecs = sentiment_vecs.masked_fill(mask.unsqueeze(-1), 0.0)
        q = self.query(sentiment_vecs)
        k = self.key(sentiment_vecs)
        v = self.value(sentiment_vecs)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, 7, 7]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        weighted = torch.matmul(attn_weights, v)  # [batch_size, num_attributes, input_dim]



        output = weighted + sentiment_vecs  # residual

        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0.0)
            valid_counts = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        else:
            valid_counts = torch.tensor(self.num_attributes, dtype=torch.float32, device=probs.device).unsqueeze(
                0).repeat(batch_size, 1)

        pooled = output.sum(dim=1) / valid_counts  # [B, D]

        gate_input = torch.cat([rating_feature, pooled], dim=-1)
        gate = self.fuse_gate(gate_input)  # [B, D]
        final_rep = gate * rating_feature + (1 - gate) * pooled

        review_score = self.review_score_predictor(final_rep).squeeze(-1)  # [B]
        return review_score

    def user_forward(self, opinion_probs, rating_feature, mask=None):
        sentiment_vecs = torch.matmul(opinion_probs, self.sentiment_embedding)  # [B, 7, 448]

        if mask is not None:  ##do we need two masks??
            sentiment_vecs = sentiment_vecs.masked_fill(mask.unsqueeze(-1), 0.0)
            # sentiment_vecs_d = sentiment_vecs_d.masked_fill(mask.unsqueeze(-1), 0.0)
        # q = self.diff_query_proj(rating_diff.unsqueeze(-1)).unsqueeze(1)  # [B, 1, D]

        q = self.query(sentiment_vecs)
        k = self.key(sentiment_vecs)
        v = self.value(sentiment_vecs)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, 7, 7]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), -1e9)

        # Calculate attention scores
        # attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, num_attributes, num_attributes]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, 7, 7]

        # Apply attention to values
        weighted = torch.matmul(attn_weights, v)  # [batch_size, num_attributes, input_dim]

        output = weighted + sentiment_vecs  # residual
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0.0)
            valid_counts = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        else:
            valid_counts = torch.tensor(self.num_attributes, dtype=torch.float32,
                                        device=opinion_probs.device).unsqueeze(0).repeat(batch_size, 1)

        pooled = output.sum(dim=1) / valid_counts  # [B, D]

        gate_input = torch.cat([rating_feature, pooled], dim=-1)
        gate = self.fuse_gate(gate_input)  # [B, D]
        final_rep = gate * rating_feature + (1 - gate) * pooled

        review_score = self.review_score_predictor(final_rep).squeeze(-1)  # [B]

        return review_score


    def forward(self, opinion_probs, opinion_sum_labels, rating_feature, mask=None, use_ground_truth_labels=False, true_labels=None):
        # x shape: [batch_size, num_attributes, input_dim]
        batch_size = opinion_probs.size(0)

        sentiment_vecs = torch.matmul(opinion_sum_labels, self.sentiment_embedding)  # [B, 7, 448]

        if mask is not None:  ##do we need two masks??
            sentiment_vecs = sentiment_vecs.masked_fill(mask.unsqueeze(-1), 0.0)

        q = self.query(sentiment_vecs)
        k = self.key(sentiment_vecs)
        v = self.value(sentiment_vecs)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, 7, 7]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), -1e9)

        # Calculate attention scores
        # attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, num_attributes, num_attributes]
        attn_weights = F.softmax(attn_scores, dim=-1) #[batch, 7, 7]

        # Apply attention to values
        weighted = torch.matmul(attn_weights, v)  # [batch_size, num_attributes, input_dim]

        output = weighted + sentiment_vecs  # residual
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0.0)
            valid_counts = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        else:
            valid_counts = torch.tensor(self.num_attributes, dtype=torch.float32, device=opinion_probs.device).unsqueeze(0).repeat(batch_size, 1)

        pooled = output.sum(dim=1) / valid_counts  # [B, D]
        # output = weighted.squeeze(1) #[B,D]

        gate_input = torch.cat([rating_feature, pooled], dim=-1)
        # gate_input = torch.cat([rating_feature, output], dim=-1)
        gate = self.fuse_gate(gate_input)  # [B, D]
        final_rep = gate * rating_feature + (1 - gate) * pooled

        review_score = self.review_score_predictor(final_rep).squeeze(-1)  # [B]

        if use_ground_truth_labels and true_labels is not None:
            gt_review_score = self.gt_review_score(true_labels, rating_feature, mask)
            return review_score, gt_review_score

        else:
            return review_score, output


class MMOE_Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(MMOE_Classifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)  
        self.linear2 = nn.Linear(input_dim, num_classes)
        self.init_weights()  
    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):
        mlp_vector = F.relu(self.linear1(hidden))  
        logits = self.linear2(mlp_vector)          
        return F.softmax(logits, dim=-1)           

class MultiTaskModel(nn.Module):
    def __init__(self, num_users, num_items, input_dim, expert_dim, num_experts, sentiment_loss, num_attributes, alpha, belta):
        super(MultiTaskModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, dropout=0.5)
        self.num_attributes = num_attributes
        # self.max_label_size = max_label_size
        self.sentiment_loss = sentiment_loss
        self.belta = belta
        self.alpha = alpha

        # User & Item Embeddings
        self.user_embedding = nn.Embedding(num_users, input_dim)
        self.item_embedding = nn.Embedding(num_items, input_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        # Experts with transformer layers
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, expert_dim) for _ in range(num_experts)
        ])

        # Gate for rating prediction
        self.rating_gate = GateNetwork(num_experts, input_dim)

        # Gates for review generation (one per attribute)
        self.review_gates = nn.ModuleList([
            GateNetwork(num_experts, input_dim) for _ in range(num_attributes)
        ])

        # User-Item Interaction Layer
        self.interaction_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(input_dim)
            nn.LayerNorm(input_dim)
        )

        # Attribute towers for review generation
        self.attribute_towers = nn.ModuleList([
            MMOE_Classifier(input_dim)
            for _ in range(num_attributes)
        ])

        self.rating_tower = MMOE_MLP(input_dim)

        # Attention Network for review score
        # self.embedding_transform = nn.Linear(input_dim, expert_dim)
        # Review attention network
        self.review_attention = ReviewAttentionNetwork(input_dim, num_attributes)

        # Review score prediction
        self.review_score = nn.Sequential(
            nn.Linear(24, 64),  # 7*4=28 6*4 = 24
            nn.GELU(),
            nn.Linear(64, 1)
        )



    def forward(self, user_ids, item_ids, group_ids, ratings, opinion_labels, opinion_sum_labels, processed_labels, is_predicted_opinions=True):
        # === Convert User-Item Indices to Embeddings ===
        group_lengths = []
        for b, group in enumerate(group_ids):
            group_lengths.append(len(group))
        user_emb = self.user_embedding(user_ids.unsqueeze(0))  # Shape: `[1, batch_size, emb_dim]`
        item_emb = self.item_embedding(item_ids.unsqueeze(0))  # Shape: `[1, batch_size, emb_dim]`

        group_emb = [self.user_embedding(torch.tensor(g, device=item_ids.device)).mean(dim=0) for g in
                     group_ids]  # list of [D]
        group_emb = torch.stack(group_emb, dim=0).unsqueeze(0) #[1, batch_size,448]


        user_item_representation = torch.cat([user_emb, item_emb], dim=0)  # [2, batch, 448]
        group_item_representation = torch.cat([group_emb, user_emb], dim=0)
        #user-item interaction
        user_item_rep = self.interaction_layer(user_item_representation) #[2,512,512]
        group_item_rep = self.interaction_layer(group_item_representation)


        user_item_rep = self.pos_encoder(user_item_rep)  # [2, batch_sie, 512]
        group_item_rep = self.pos_encoder(group_item_rep)
        mean_pool = user_item_rep.mean(dim=0)  # [batch, emsize]
        group_mean_pool = group_item_rep.mean(dim=0)

        #==============user-item for rating prediction============
        expert_outputs = [expert(mean_pool) for expert in self.experts]  # list 3: Each: [batch_size, expert_dim]

        # # Stack and reshape back
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, expert_dim]
        # Rating Prediction Task
        rating_gate_weights = self.rating_gate(mean_pool)  # torch.Size([512, 5])
        rating_expert_output = torch.einsum("be,bed->bd", rating_gate_weights, expert_outputs) #[batch_size, exp_dim]
        predicted_rating_score = self.rating_tower(rating_expert_output) #[batch_size,1]
        # print(predicted_rating_score)


        if is_predicted_opinions:
            user_logits_list = []
            user_embeddings_list = []
            #===========user-item prediction=============
            for i, gate in enumerate(self.review_gates):
                gate_weights = gate(mean_pool)  # user-item interaction

                attribute_expert_output = torch.einsum("be,bed->bd", gate_weights, expert_outputs) #[batch, 448]



                logits = self.attribute_towers[i](attribute_expert_output) #[512, 4]


                user_logits_list.append(logits)
                user_embeddings_list.append(attribute_expert_output)
            user_stacked_probs = torch.stack(user_logits_list, dim=1)
            user_attribute_probs = F.softmax(user_stacked_probs, dim=-1)  # predicted user-item one-hot labels

            #==============group-item prediction==============
            group_logits_list = []
            group_embeddings_list = []
            gp_expert_outputs = [expert(group_mean_pool) for expert in self.experts]  # list 3: Each: [batch_size, expert_dim]

            # # Stack and reshape back
            gp_expert_outputs = torch.stack(gp_expert_outputs, dim=1)  # [batch_size, num_experts, expert_dim]

            for i, gate in enumerate(self.review_gates):
            # gate_weights = gate(group_item)
                gate_weights = gate(group_mean_pool)  # user-item interaction

                attribute_expert_output = torch.einsum("be,bed->bd", gate_weights, gp_expert_outputs) #[batch, 448]



                logits = self.attribute_towers[i](attribute_expert_output) #[512, 4]

            # logits_list.append(F.pad(logits, (0, max_label_size - logits.size(1)), mode="constant", value=0))
                group_logits_list.append(logits)
                group_embeddings_list.append(attribute_expert_output)


            group_stacked_probs = torch.stack(group_logits_list, dim=1)

            pred_group_labels = F.softmax(group_stacked_probs, dim=-1)
            group_lengths = torch.tensor(group_lengths, device=item_ids.device, dtype=torch.float32)  # [B]
            predicted_opinion_sum_labels = pred_group_labels * group_lengths.view(-1, 1, 1)  # [B, 7, 4]

            opinion_mask = (processed_labels == -1)


            pred_user_score = self.review_attention.user_forward(
                opinion_probs=user_attribute_probs,
                rating_feature=rating_expert_output,
                mask=opinion_mask,  ##need to be revised: this mask is from gt_group_labels, not user_opinion_label(6.20)
            )


            user_final_score = predicted_rating_score + 0.8 * pred_user_score


            predicted_review_score, gt_review_score = self.review_attention(
                opinion_probs=pred_group_labels,
                opinion_sum_labels=predicted_opinion_sum_labels,
                rating_feature=rating_expert_output,
                true_labels=opinion_sum_labels, #gt_group_labels
                mask=opinion_mask,
                use_ground_truth_labels=True  
            )

            final_score = predicted_rating_score + 0.8 * predicted_review_score
        else:
            attribute_probs = None
            predicted_review_score = torch.zeros_like(predicted_rating_score)
            final_score = predicted_rating_score


        return predicted_rating_score, pred_group_labels, user_attribute_probs, user_final_score, predicted_review_score, final_score, gt_review_score, predicted_opinion_sum_labels

    def calculate_loss(self, user_ids, item_ids, group_ids, ratings, gt_user_labels, group_sum_labels, gt_group_labels, group_processed_labels, is_predicted_opinions=True):
        ##we need compare performance when using user-item embs and group-item embs
        #predicted_ratings: predicted from user-item embedding;
        # predicted_opinions: predicted labels from group embeddings
        #predicted_review_score: score from predicted lables through attention network
        #final_score: final ratings adjusted by predicted opinions
        #gt_review_score: score from true labels trhough attention networks
        predicted_ratings, predicted_opinions, predicted_user_opinions, _, predicted_review_score, final_score, gt_review_score, pred_opinion_sum_labels = self.forward(user_ids, item_ids, group_ids, ratings, gt_group_labels, group_sum_labels, group_processed_labels, is_predicted_opinions)
        rating_loss = F.mse_loss(predicted_ratings, ratings)
        # print("Any NaN in target:", torch.isnan(final_score).any().item())

        if is_predicted_opinions:
            rating_review_loss = F.mse_loss(final_score, ratings)
            device = user_ids.device
            user_opinion_loss = 0.0
            group_opinion_loss = 0.0
            valid_opinion_count = 0
            lambda_user = 0.4

            for i in range(self.num_attributes):
                
                curr_user_labels = gt_user_labels[:, i, :]
                curr_group_labels = gt_group_labels[:, i, :]

                user_valid_mask = (curr_user_labels.sum(dim=1)) > 0  # [batch_size]
                group_valid_mask = (curr_group_labels.sum(dim=1)) > 0

                if user_valid_mask.any():
                    user_valid_pred = predicted_opinions[:, i, :][user_valid_mask]
                    group_valid_pred = predicted_opinions[:, i, :][group_valid_mask]
                    user_valid_true = torch.argmax(curr_user_labels[user_valid_mask], dim=1)
                    group_valid_true = torch.argmax(curr_group_labels[group_valid_mask], dim=1)

                    user_opinion_loss += F.cross_entropy(user_valid_pred, user_valid_true)
                    group_opinion_loss += F.cross_entropy(group_valid_pred, group_valid_true)
                    valid_opinion_count += 1

            if valid_opinion_count > 0:
                user_opinion_loss /= valid_opinion_count
                group_opinion_loss /= valid_opinion_count
                opinion_loss = (1 - lambda_user) * user_opinion_loss + lambda_user * group_opinion_loss


            else:
                opinion_loss = 0.0


            #===================sentiment loss============
            if self.sentiment_loss > 0:

                sentiment_vec_pred = torch.matmul(predicted_opinions,
                                                  self.review_attention.sentiment_embedding)  # [B, 7, D]
                sentiment_vec_gt = torch.matmul(gt_group_labels.float(),
                                                self.review_attention.sentiment_embedding)  # [B, 7, D]
                sentiment_vec_sum = torch.matmul(group_sum_labels.float(),
                                                self.review_attention.sentiment_embedding)  # [B, 7, D]

                valid_mask = (gt_group_labels.sum(dim=-1) > 0)  # [B, 7]
                cosine_sum_loss = F.cosine_embedding_loss(sentiment_vec_pred[valid_mask],
                    sentiment_vec_sum[valid_mask],
                    torch.ones(valid_mask.sum(), device=sentiment_vec_gt.device))

                #final
                total_loss = rating_review_loss + self.alpha * opinion_loss + self.belta * cosine_sum_loss


            else:
                total_loss = rating_review_loss

    def rating_forward(self, user_ids, item_ids):
        # === Convert User-Item Indices to Embeddings ==
        user_emb = self.user_embedding((user_ids.unsqueeze(0)))  # Shape: `[1, batch_size, emb_dim]`
        item_emb = self.item_embedding((item_ids.unsqueeze(0)))
        user_item_representation = torch.cat([user_emb, item_emb], dim=0)  # [2, batch, emsize]
        user_item_rep = self.interaction_layer(user_item_representation)  # [2,512,448]

        # ==========================================================================
        # Apply Transformer Encoder

        user_item_rep = self.pos_encoder(user_item_rep)  # [2, batch_sie, 448]

        mean_pool = user_item_rep.mean(dim=0)  # [batch, emsize]

        expert_outputs = [expert(mean_pool) for expert in self.experts]  # list 5: Each: [batch_size, expert_dim]
        # # Stack and reshape back
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, expert_dim]

        # Rating Prediction Task
        rating_gate_weights = self.rating_gate(mean_pool)  # torch.Size([512, 5])
        rating_expert_output = torch.einsum("be,bed->bd", rating_gate_weights, expert_outputs)  # [batch_size, exp_dim]
        predicted_rating_score = self.rating_tower(rating_expert_output)  # [batch_size,1]
        return predicted_rating_score

    def predict_opinions(self, user_ids, item_ids, group_ids, ratings, gt_user_labels, group_sum_labels, gt_group_labels, group_processed_labels):
        predicted_ratings, predicted_labels, user_pred_labels, _, predicted_review_score, final_score, gt_review_score, _ = self.forward(user_ids, item_ids, group_ids, ratings, gt_group_labels, group_sum_labels, group_processed_labels,
                                                                             is_predicted_opinions=True)
        rating_review_loss = F.mse_loss(final_score, ratings)
        rating_loss = F.mse_loss(predicted_ratings, ratings)
        device = user_ids.device
        user_opinion_loss = 0.0
        group_opinion_loss = 0.0
        valid_opinion_count = 0
        lambda_user = 0.4

        for i in range(self.num_attributes):
            curr_user_labels = gt_user_labels[:, i, :]
            curr_group_labels = gt_group_labels[:, i, :]
            
            user_valid_mask = (curr_user_labels.sum(dim=1)) > 0  # [batch_size]
            group_valid_mask = (curr_group_labels.sum(dim=1)) > 0

            if user_valid_mask.any():
                user_valid_pred = predicted_labels[:, i, :][user_valid_mask]
                group_valid_pred = predicted_labels[:, i, :][group_valid_mask]
                user_valid_true = torch.argmax(curr_user_labels[user_valid_mask], dim=1)
                group_valid_true = torch.argmax(curr_group_labels[group_valid_mask], dim=1)

                user_opinion_loss += F.cross_entropy(user_valid_pred, user_valid_true)
                group_opinion_loss += F.cross_entropy(group_valid_pred, group_valid_true)
                valid_opinion_count += 1

        if valid_opinion_count > 0:
            user_opinion_loss /= valid_opinion_count
            group_opinion_loss /= valid_opinion_count
            opinion_loss = (1 - lambda_user) * user_opinion_loss + lambda_user * group_opinion_loss

        else:
            opinion_loss = 0.0

            total_loss = rating_review_loss + self.alpha * opinion_loss



        else:
            total_loss = rating_review_loss + opinion_loss
        return rating_review_loss, total_loss, predicted_labels, user_pred_labels #[batch_size, max_item_num]

    def predict(self, user_ids, item_ids):


        predicted_ratings = self.rating_forward(user_ids, item_ids)
        # predicted_ratings, _, _, _ = self.forward(user_ids, item_ids, opinion_weights)
        return predicted_ratings #[batch_size, max_item_num]

    def predict_rating_with_reviews(self, user_ids, item_ids, group_ids, ratings, opinion_labels, group_sum_labels, processed_labels, is_predicted_opinions=True):
        predicted_ratings, group_pred_labels, user_pred_labels, user_final_score, _, final_score, _, _ = self.forward(user_ids, item_ids, group_ids, ratings, opinion_labels, group_sum_labels, processed_labels, is_predicted_opinions=True)
        return predicted_ratings, group_pred_labels, user_pred_labels, user_final_score, final_score

