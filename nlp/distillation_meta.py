import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter
from functional_forward_bert import functional_bert_for_classification
from transformers import PreTrainedModel


class MetaPatientDistillation(nn.Module):
    def __init__(self, t_config, s_config):
        super(MetaPatientDistillation, self).__init__()
        self.t_config = t_config
        self.s_config = s_config

    def forward(self, t_model, s_model, order, input_ids, token_type_ids, attention_mask, labels, args, teacher_grad):
        if teacher_grad:
            t_outputs = t_model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        else:
            with torch.no_grad():
                t_outputs = t_model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

        if isinstance(s_model, PreTrainedModel):
            s_outputs = s_model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                labels=labels)
        else:
            s_outputs = functional_bert_for_classification(
                s_model,
                self.s_config,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        t_logits, t_features = t_outputs[0], t_outputs[-1]
        train_loss, s_logits, s_features = s_outputs[0], s_outputs[1], s_outputs[-1]

        if args.logits_mse:
            soft_loss = F.mse_loss(t_logits, s_logits)
        else:
            T = args.temperature
            soft_targets = F.softmax(t_logits / T, dim=-1)

            probs = F.softmax(s_logits / T, dim=-1)
            soft_loss = F.mse_loss(soft_targets, probs) * T * T

        if args.beta == 0:  # if beta=0, we don't even compute pkd_loss to save some time
            pkd_loss = torch.zeros_like(soft_loss)
        else:
            t_features = torch.cat(t_features[1:-1], dim=0).view(self.t_config.num_hidden_layers - 1,
                                                                 -1,
                                                                 args.max_seq_length,
                                                                 self.t_config.hidden_size)[:, :, 0]

            s_features = torch.cat(s_features[1:-1], dim=0).view(self.s_config.num_hidden_layers - 1,
                                                                 -1,
                                                                 args.max_seq_length,
                                                                 self.s_config.hidden_size)[:, :, 0]

            t_features = itemgetter(order)(t_features)
            t_features = t_features / t_features.norm(dim=-1).unsqueeze(-1)
            s_features = s_features / s_features.norm(dim=-1).unsqueeze(-1)
            pkd_loss = F.mse_loss(s_features, t_features, reduction="mean")

        return train_loss, soft_loss, pkd_loss

    def s_prime_forward(self, s_prime, input_ids, token_type_ids, attention_mask, labels, args):

        s_outputs = functional_bert_for_classification(
            s_prime,
            self.s_config,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
            is_train=False
        )

        train_loss, s_logits, s_features = s_outputs[0], s_outputs[1], s_outputs[-1]

        return train_loss
