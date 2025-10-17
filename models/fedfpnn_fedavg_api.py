import copy
import random
import numpy as np
import torch
## import wandb  # Disabled for offline run
from data_process.dataset import FedDatasetCV
from models.client import Client, FedFPNNClient
from models.fpnn import *


class FedAvgAPI(object):
    def __init__(self, dataset: FedDatasetCV, global_model, p_args):
        self.args = p_args
        train_data_global, test_data_global, train_data_local_dict, test_data_local_dict = \
            dataset.get_local_data()
        self.test_global = test_data_global
        self.val_global = None

        self.client_list = []
        self.client_rule_list = []
        _, self.train_data_local_class_count = dataset.get_class_cound_list()
        _, self.train_data_local_num_dict = dataset.get_local_saml_num()
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.global_model = global_model

        self._setup_clients(self.train_data_local_num_dict, self.train_data_local_class_count, train_data_local_dict,
                            test_data_local_dict)
        self.rules_client_dict = {}

    def _setup_clients(self, train_data_local_num_dict, train_data_local_class_count,
                       train_data_local_dict, test_data_local_dict):
        self.args.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.n_client):
            if client_idx not in train_data_local_dict or client_idx not in test_data_local_dict:
                print(f"[WARNING] Creating dummy client {client_idx} (missing train or test DataLoader)")
                # Create a dummy client with zero samples and empty DataLoaders
                class DummyClient:
                    def __init__(self, n_rule_limit):
                        self.local_rule_idxs = []
                        self.rules_idx_list = []
                        self.local_sample_number = 0
                        self.n_rule_limit = n_rule_limit
                        self.local_class_count = {}
                    def train(self, w_global):
                        return w_global
                    def local_test(self, is_test):
                        # return consistent metrics structure
                        return {'test_total': 0, 'test_correct': 0, 'test_loss': 0.0, 'fs': torch.empty(0, self.n_rule_limit)}
                self.client_rule_list.append([])
                self.client_list.append(DummyClient(self.args.n_rule))
                continue
            local_rules_idx_list = copy.deepcopy(self.global_model.rules_idx_list)
            c = FedFPNNClient(client_idx, local_rules_idx_list,
                              train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                              train_data_local_num_dict[client_idx], train_data_local_class_count[client_idx],
                              self.args, self.global_model)
            self.client_rule_list.append(self.global_model.rules_idx_list)
            self.client_list.append(c)
        self.args.logger.info("############setup_clients (END)#############")

    def get_rules_client_dict(self):
        for rule_i in range(self.args.n_rule):
            self.rules_client_dict[rule_i] = []
            for client_j in range(self.args.n_client):
                if rule_i in self.client_list[client_j].local_rule_idxs:
                    self.rules_client_dict[rule_i].append(client_j)

    def train(self):
        w_global = self.global_model.cpu().state_dict()
        metrics_list = []
        # Prepare tensors to record global test accuracy and loss
        self.global_test_acc_tsr = torch.zeros(self.args.comm_round)
        self.global_test_loss_tsr = torch.zeros(self.args.comm_round)
        for round_idx in range(self.args.comm_round):
            self.args.logger.info("################Communication round : {}".format(round_idx))
            w_locals = []
            # for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            # Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            client_indexes_per_round = self._client_sampling(round_idx, self.args.n_client,
                                                             self.args.n_client_per_round)
            self.args.logger.info("client_indexes = " + str(client_indexes_per_round))
            for client_idx in client_indexes_per_round:
                client = self.client_list[client_idx]
                # train on local client (dummy client returns w_global)
                w = client.train(w_global)
                w_locals.append((getattr(client, 'local_sample_number', 0), copy.deepcopy(w)))
            # update global weights
            w_global = self._aggregate(w_locals)
            self.global_model.update_model(w_global)
            # test results
            metrics_rtn = self._local_test_on_all_clients(round_idx)
            metrics_list.append(metrics_rtn)
            # Record global test accuracy and loss for this round
            self.global_test_acc_tsr[round_idx] = metrics_rtn.get('test_acc', 0.0)
            self.global_test_loss_tsr[round_idx] = metrics_rtn.get('test_loss', 0.0)
        return metrics_list

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.args.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        averaged_params = self.global_model.cpu().state_dict()
        # aggregate the rule parameters and other params robustly
        for k in averaged_params.keys():
            if "rule" in k:
                rule_training_num = 0
                for idx in range(len(w_locals)):
                    (sample_num, _) = w_locals[idx]
                    # use client_list to check whether this client owns the rule
                    if int(k.split(".")[0].split("_")[1]) in self.client_list[idx].rules_idx_list:
                        rule_training_num += sample_num
                if rule_training_num == 0:
                    # Avoid division by zero, skip aggregation for this rule
                    continue
                tag_j = 0
                for idx in range(len(w_locals)):
                    local_sample_number, local_model_params = w_locals[idx]
                    if int(k.split(".")[0].split("_")[1]) in self.client_list[idx].rules_idx_list:
                        w = local_sample_number / rule_training_num
                        if tag_j == 0:
                            averaged_params[k] = local_model_params[k] * w
                            tag_j = 1
                        else:
                            averaged_params[k] += local_model_params[k] * w
            else:
                training_num_fs = 0
                for i in range(0, len(w_locals)):
                    local_sample_number, _ = w_locals[i]
                    training_num_fs += local_sample_number
                if training_num_fs == 0:
                    # nothing to aggregate, keep global param
                    continue
                for i in range(0, len(w_locals)):
                    local_sample_number_fs, local_model_params = w_locals[i]
                    w = local_sample_number_fs / training_num_fs
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _safe_one_hot_count(self, fs_tensor):
        """
        Given an (N, n_rule) fs_tensor, compute count per rule of argmax over dim=-1.
        Returns a long tensor of shape (n_rule,) on the configured device.
        Handles empty tensors gracefully.
        """
        if fs_tensor is None or fs_tensor.numel() == 0:
            return torch.zeros(self.args.n_rule, dtype=torch.long).to(self.args.device)
        _, idx = torch.max(fs_tensor, -1)
        # ensure idx is at least 1-D
        if idx.numel() == 0:
            return torch.zeros(self.args.n_rule, dtype=torch.long).to(self.args.device)
        counts = torch.nn.functional.one_hot(idx, num_classes=self.args.n_rule).sum(0)
        return counts.long().to(self.args.device)

    def _local_test_on_all_clients(self, round_idx):

        self.args.logger.info("################local_test_on_all_clients : {}".format(round_idx))

        metrics = {
            'train_num_samples': [],
            'train_num_correct': [],
            'train_losses': [],
            'test_num_samples': [],
            'test_num_correct': [],
            'test_losses': [],
        }

        train_rule_count_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)
        test_rule_count_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)
        train_rule_contr_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)
        test_rule_contr_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)

        train_acc_local = torch.zeros(self.args.n_client).to(self.args.device)
        test_acc_local = torch.zeros(self.args.n_client).to(self.args.device)
        train_loss_local = torch.zeros(self.args.n_client).to(self.args.device)
        test_loss_local = torch.zeros(self.args.n_client).to(self.args.device)

        train_rule_fs = torch.empty(0, self.args.n_rule).to(self.args.device)
        test_rule_fs = torch.empty(0, self.args.n_rule).to(self.args.device)

        for client_idx in range(self.args.n_client):
            # If missing test loader for this client, produce zero metrics but keep client slot
            if client_idx not in self.test_data_local_dict or self.test_data_local_dict.get(client_idx) is None:
                print(f"[WARNING] Skipping metrics for client {client_idx} (no test DataLoader)")
                metrics['train_num_samples'].append(0)
                metrics['train_num_correct'].append(0)
                metrics['train_losses'].append(0)
                train_acc_local[client_idx] = 0
                train_loss_local[client_idx] = 0
                # leave corresponding rule contributions/counts as zeros
                continue

            client = self.client_list[client_idx]

            # TRAIN (we call client's local_test(False) which returns training-set metrics per your code)
            train_local_metrics = client.local_test(False)
            train_num_client = int(copy.deepcopy(train_local_metrics.get('test_total', 0)))
            train_correct_num_client = int(copy.deepcopy(train_local_metrics.get('test_correct', 0)))
            train_loss_all_client = float(copy.deepcopy(train_local_metrics.get('test_loss', 0.0)))
            metrics['train_num_samples'].append(train_num_client)
            metrics['train_num_correct'].append(train_correct_num_client)
            metrics['train_losses'].append(train_loss_all_client)

            if train_num_client == 0:
                train_acc_client = 0.0
                train_loss_client = 0.0
            else:
                train_acc_client = train_correct_num_client / train_num_client
                train_loss_client = train_loss_all_client / train_num_client

            train_acc_local[client_idx] = train_acc_client
            train_loss_local[client_idx] = train_loss_client

            train_rule_fs_client = copy.deepcopy(train_local_metrics.get('fs', torch.empty(0, self.args.n_rule)))
            if train_rule_fs_client is None:
                train_rule_fs_client = torch.empty(0, self.args.n_rule).to(self.args.device)
            if train_rule_fs_client.numel() > 0:
                train_rule_fs = torch.cat((train_rule_fs, train_rule_fs_client), 0)
            # safe counts and contributions
            train_rule_count_client = self._safe_one_hot_count(train_rule_fs_client)
            train_rule_contr_client = train_rule_fs_client.mean(0) if train_rule_fs_client.numel() > 0 else torch.zeros(self.args.n_rule).to(self.args.device)
            train_rule_count_local[client_idx, :] = train_rule_count_client
            train_rule_contr_local[client_idx, :] = train_rule_contr_client

            # TEST
            test_local_metrics = client.local_test(True)
            test_num_client = int(copy.deepcopy(test_local_metrics.get('test_total', 0)))
            test_correct_num_client = int(copy.deepcopy(test_local_metrics.get('test_correct', 0)))
            test_loss_all_client = float(copy.deepcopy(test_local_metrics.get('test_loss', 0.0)))
            metrics['test_num_samples'].append(test_num_client)
            metrics['test_num_correct'].append(test_correct_num_client)
            metrics['test_losses'].append(test_loss_all_client)

            if test_num_client == 0:
                print(f"[WARNING] Skipping test for client {client_idx} due to zero test samples.")
                test_acc_client = 0.0
                test_loss_client = 0.0
            else:
                test_acc_client = test_correct_num_client / test_num_client
                test_loss_client = test_loss_all_client / test_num_client

            test_acc_local[client_idx] = test_acc_client
            test_loss_local[client_idx] = test_loss_client

            test_rule_fs_client = copy.deepcopy(test_local_metrics.get('fs', torch.empty(0, self.args.n_rule)))
            if test_rule_fs_client is None:
                test_rule_fs_client = torch.empty(0, self.args.n_rule).to(self.args.device)
            if test_rule_fs_client.numel() > 0:
                test_rule_fs = torch.cat((test_rule_fs, test_rule_fs_client), 0)

            test_rule_count_client = self._safe_one_hot_count(test_rule_fs_client)
            test_rule_contr_client = test_rule_fs_client.mean(0) if test_rule_fs_client.numel() > 0 else torch.zeros(self.args.n_rule).to(self.args.device)
            test_rule_count_local[client_idx, :] = test_rule_count_client
            test_rule_contr_local[client_idx, :] = test_rule_contr_client

            # Update client's rule list based on training contribution (keep existing logic but guarded)
            if train_rule_contr_client.numel() > 0 and len(client.rules_idx_list) > 0:
                c_th = np.arange(self.args.n_rule)[train_rule_contr_client.cpu() > 1 / max(1, len(client.rules_idx_list))]
                if len(client.rules_idx_list) >= client.n_rule_limit:
                    client.update_rule_idx_list(c_th)

        # After iterating all clients compute global aggregates safely
        # global train/test rule counts from concatenated fs matrices
        train_rule_count = self._safe_one_hot_count(train_rule_fs) if train_rule_fs.numel() > 0 else torch.zeros(self.args.n_rule, dtype=torch.long).to(self.args.device)
        test_rule_count = self._safe_one_hot_count(test_rule_fs) if test_rule_fs.numel() > 0 else torch.zeros(self.args.n_rule, dtype=torch.long).to(self.args.device)

        train_rule_contr = train_rule_fs.mean(0) if train_rule_fs.numel() > 0 else torch.zeros(self.args.n_rule).to(self.args.device)
        test_rule_contr = test_rule_fs.mean(0) if test_rule_fs.numel() > 0 else torch.zeros(self.args.n_rule).to(self.args.device)

        metrics_rule = {}
        for rule_idx in torch.arange(self.args.n_rule):
            metrics_rule[f"rule{rule_idx + 1}_count"] = int(train_rule_count[rule_idx])
            metrics_rule[f"rule{rule_idx + 1}_contr"] = float(train_rule_contr[rule_idx])
            for client_idx in range(self.args.n_client):
                metrics_rule[f"client{client_idx + 1}_rule{rule_idx + 1}_count"] = int(
                    train_rule_count_local[client_idx, rule_idx])
                metrics_rule[f"client{client_idx + 1}_rule{rule_idx + 1}_contr"] = float(
                    train_rule_contr_local[client_idx, rule_idx])
                metrics_rule[f"client{client_idx + 1}_train_acc"] = float(
                    train_acc_local[client_idx])
                metrics_rule[f"client{client_idx + 1}_train_loss"] = float(
                    train_loss_local[client_idx])
                metrics_rule[f"client{client_idx + 1}_test_acc"] = float(
                    test_acc_local[client_idx])
                metrics_rule[f"client{client_idx + 1}_test_loss"] = float(
                    test_loss_local[client_idx])

        # test on training dataset (safe division)
        total_train_samples = sum(metrics['train_num_samples']) if len(metrics['train_num_samples']) > 0 else 0
        total_train_correct = sum(metrics['train_num_correct']) if len(metrics['train_num_correct']) > 0 else 0
        train_acc = total_train_correct / total_train_samples if total_train_samples > 0 else 0.0
        train_loss = sum(metrics['train_losses']) / total_train_samples if total_train_samples > 0 else 0.0

        # test on test dataset (safe division)
        total_test_samples = sum(metrics['test_num_samples']) if len(metrics['test_num_samples']) > 0 else 0
        total_test_correct = sum(metrics['test_num_correct']) if len(metrics['test_num_correct']) > 0 else 0
        test_acc = total_test_correct / total_test_samples if total_test_samples > 0 else 0.0
        test_loss = sum(metrics['test_losses']) / total_test_samples if total_test_samples > 0 else 0.0

        metrics_stats = {'training_acc': train_acc, 'training_loss': train_loss,
                         'test_acc': test_acc, 'test_loss': test_loss}

        metrics_rtn = {**metrics_stats, **metrics_rule}
        self.args.logger.info(metrics_stats)
        for client_idx in range(self.args.n_client):
            self.args.logger.info(f"client {client_idx} ==> training_acc: {train_acc_local[client_idx]}, "
                                  f"training_loss: {train_loss_local[client_idx]}, "
                                  f"test_acc: {test_acc_local[client_idx]}, "
                                  f"test_loss: {test_loss_local[client_idx]}")
        return metrics_rtn

    def _eval_rules_on_all_clients(self, round_idx):
        # For brevity re-use the robust testing logic in _local_test_on_all_clients but only for evaluation
        return self._local_test_on_all_clients(round_idx)
