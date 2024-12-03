import copy

import numpy as np
import torch
import time
import json
from src.models.client import Client
from src.models.worker import Worker
import time
import os


class Logger():
    def __init__(self, file):
        self.file = file
        self.out_dict = {}

    def log(self, round_i ,new_dict):
        self.out_dict[round_i]= new_dict 

    def dump(self):
        with open(self.file,'w') as f:
            f.write(json.dumps(self.out_dict,indent=2))


class BaseTrainer(object):

    def __init__(self, options,dataset, model=None, optimizer=None, result_dir='results'):
        self.worker = Worker(model, optimizer, options,)
        print('>>> Activate a worker for training')

        self.options = options
        self.gpu = options['gpu']
        self.batch_size = options['batch_size']
        self.all_train_data_num = 0
        self.clients = self.setup_clients(dataset)
        assert len(self.clients) > 0
        # print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.num_round = options['num_round'] # total number of communication rounds
        self.clients_per_round = options['clients_per_round'] # useful for fedavg.

        # Initialize system metrics
        self.print_result = not options['noprint']
        self.latest_model = self.worker.get_flat_model_params()

        # logger 
        hash_tag = hash(time.time())
        hash_tag = str(hash_tag)
        hash_tag = os.path.join(result_dir, hash_tag)
        os.makedirs(hash_tag)
        self.logger = Logger(os.path.join(hash_tag,'log.json'))
        with open(os.path.join(hash_tag, 'options.json'), 'w') as f:
            json.dump(options, f, indent=2)


    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = 0 if 'device' not in options else options['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def setup_clients(self, dataset):
        """Instantiates clients based on given train and test data directories

        Returns:
            all_clients: List of clients
        """
        users, groups, train_data, test_data, avail_prob_dict = dataset
        if len(groups) == 0:
            groups = [None for _ in users]


        all_clients = []
        for user, group in zip(users, groups):
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])

            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            c = Client(user_id, group, avail_prob_dict[user],train_data[user], test_data[user], self.batch_size, self.worker,)
            all_clients.append(c)
        return all_clients

    def train(self):
        """The whole training procedure

        No returns. All results all be saved.
        """
        raise NotImplementedError

    def get_avail_clients(self, seed=1):
        """Selects num_clients clients weighted by number of samples from possible_clients

        Args:
            seed: random seed
            the availabity of clients is determined by another procedure. 
            
        Return:
            list of available clients.
        """
        np.random.seed(seed * (self.options['seed'] + 1))
        avail_client_list = []
        avail_counting = []
        counting_inFive=0
        for c in self.clients:
          if counting_inFive<250:
            p = c.available_probability
            coin = np.random.rand()
            # print(coin, "this is coin and this is p,", p )
            if coin < p:
                avail_client_list.append(c)
                counting_inFive = counting_inFive + 1

        if(len(avail_client_list)==0):
            avail_client_list.append(self.clients[0])

        # for c in self.clients:
        #     avail_client_list.append(c)
        # print(len(avail_client_list))
        return avail_client_list



    def local_train_last_round(self, round_i, selected_clients,model_name,algo,c_number,dir, **kwargs):

        solns = []  # Buffer for receiving client solutions
        stats = []# Buffer for receiving client communication costs
        loss_s=[]
        acc_s=[]

        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat = c.local_train()

            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% |".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['loss'], stat['acc']*100, ))


            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)
            loss_s.append(stat['loss'])
            acc_s.append(stat['acc']*100)

        path=dir+"_"+model_name+"_"+algo+"_"+c_number+".txt"
        np.savetxt(path, acc_s,fmt='%.02f')

        b=sorted(acc_s)
        path2=model_name+"_"+algo+"_"+c_number+"_fairness.txt"
        fh = open(path2, 'w')
        fh.write("this is for acc 10th percentile:" +str(np.percentile(b, 10))+"\r\n"
                 "50th percentile:" +str(np.percentile(b, 50))+"\r\n"
                 "90th percentile: "+str(np.percentile(b, 90))+"\r\n")
        fh.write("this is mean for acc"+ str(np.mean(acc_s))+"\r\n")
        fh.write("this is var for acc" + str(np.var(acc_s))+"\r\n")
        fh.write("difference between maximum and minimum for acc"+ str(b[len(acc_s)-1]-b[0])+"\r\n")

        fh.close()
        print('this is for acc 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (
                  np.percentile(b, 10),
                  np.percentile(b, 50),
                  np.percentile(b, 90)))

        print("this is mean for acc", np.mean(acc_s))
        print("this is var for acc", np.var(acc_s))
        print("difference between maximum and minimum for acc", b[len(acc_s)-1]-b[0])

        return solns, stats

    def local_test_last_round(self, round_i, selected_clients, model_name, algo, c_number, dir, **kwargs):
        print("最后测试")
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        loss_s = []
        acc_s = []

        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat = c.local_test()

            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% |".format(
                    round_i, c.cid, i, self.clients_per_round,
                    stat['loss'], stat['acc'] * 100, ))

            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)
            loss_s.append(stat['loss'])
            acc_s.append(stat['acc'] * 100)

        path = dir + "_" + model_name + "_" + algo + "_" + c_number + ".txt"
        np.savetxt(path, acc_s, fmt='%.02f')

        b = sorted(acc_s)
        path2 = model_name + "_" + algo + "_" + c_number + "_fairness.txt"
        fh = open(path2, 'w')
        fh.write("this is for acc 10th percentile:" + str(np.percentile(b, 10)) + "\r\n"
                                                                                  "50th percentile:" + str(
            np.percentile(b, 50)) + "\r\n"
                                    "90th percentile: " + str(np.percentile(b, 90)) + "\r\n")
        fh.write("this is mean for acc" + str(np.mean(acc_s)) + "\r\n")
        fh.write("this is var for acc" + str(np.var(acc_s)) + "\r\n")
        fh.write("difference between maximum and minimum for acc" + str(b[len(acc_s) - 1] - b[0]) + "\r\n")

        fh.close()
        print('this is for acc 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (
                  np.percentile(b, 10),
                  np.percentile(b, 50),
                  np.percentile(b, 90)))

        print("this is mean for acc", np.mean(acc_s))
        print("this is var for acc", np.var(acc_s))
        print("difference between maximum and minimum for acc", b[len(acc_s) - 1] - b[0])

        return solns, stats


    def local_train(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training, used for logging only
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs

        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat = c.local_train()

            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% |".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['loss'], stat['acc']*100, ))


            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats


    # def local_train_fold(self, round_i, selected_clients,ann,server, **kwargs):
    #     solns = []  # Buffer for receiving client solutions
    #     stats = []  # Buffer for receiving client communication costs
    #     x = copy.deepcopy(ann)
    #     for i, c in enumerate(selected_clients, start=1):
    #         # Communicate the latest model
    #         c.set_flat_model_params(self.latest_model)
    #
    #         # Solve minimization locally
    #         soln, stat,step = c.local_train_fold(ann,server)
    #
    #         if self.print_result:
    #             print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
    #                   "Loss {:>.4f} | Acc {:>5.2f}% |".format(
    #                    round_i, c.cid, i, self.clients_per_round,
    #                    stat['loss'], stat['acc']*100, ))
    #         solns.append(soln)
    #         stats.append(stat)
    #     temp = {}
    #     for k, v in ann.named_parameters():
    #         temp[k] = v.data.clone()
    #
    #     for k, v in x.named_parameters():
    #
    #         local_steps = step
    #
    #         device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #         v.data = v.data.to(device)
    #         temp[k] =  temp[k].to(device)
    #         ann.control[k]=ann.control[k].to(device)
    #         x.control[k]=x.control[k].to(device)
    #
    #         ann.control[k] = ann.control[k] - server.control[k] + (v.data - temp[k]) / (local_steps * self.optimizer.lr)
    #         ann.delta_y[k] = temp[k] - v.data
    #
    #         ann.delta_control[k] = ann.control[k] - x.control[k]
    #
    #
    #     return solns, stats, ann

    def local_train_fold2(self, round_i, selected_clients, ann, server, **kwargs):
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs

        for i, c in enumerate(selected_clients, start=1):
            x = copy.deepcopy(ann[c.cid])
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat, step = c.local_train_fold(ann[c.cid], server)

            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% |".format(
                    round_i, c.cid, i, self.clients_per_round,
                    stat['loss'], stat['acc'] * 100, ))
            solns.append(soln)
            stats.append(stat)
            temp = {}
            for k, v in ann[c.cid].named_parameters():
               temp[k] = v.data.clone()

            for k, v in x.named_parameters():
              local_steps = step

              device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
              v.data = v.data.to(device)
              temp[k] = temp[k].to(device)
              ann[c.cid].control[k] = ann[c.cid].control[k].to(device)
              x.control[k] = x.control[k].to(device)
              server.control[k]=server.control[k].to(device)

              ann[c.cid].control[k] = ann[c.cid].control[k] - server.control[k] + (v.data - temp[k]) / (local_steps*self.optimizer.lr)#self.optimizer.lr
              ann[c.cid].delta_y[k] = temp[k] - v.data

              ann[c.cid].delta_control[k] = ann[c.cid].control[k] - x.control[k]

        return solns, stats, ann

    def evaluate_train(self, **kwargs):
        return self.base_evaluate(eval_on_train=True)

    def evaluate_test(self, **kwargs):
        return self.base_evaluate(eval_on_train=False)

    def evaluate_train_fold(self,nn, **kwargs):
        return self.base_evaluate_fold(nn,eval_on_train=True)

    def evaluate_test_fold(self, nn,**kwargs):
        return self.base_evaluate_fold(nn,eval_on_train=False)

    def base_evaluate(self, eval_on_train ,**kwargs):
        """
            Evaluate results on training data/test data.
        """
        num_samples = 0
        total_loss = 0
        total_correct = 0
        for c in self.clients:
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Evaluate locally
            if eval_on_train:
                return_dict = c.evaluate_train(**kwargs)
            else:
                return_dict = c.evaluate_test(**kwargs)

            num_samples += return_dict["num_samples"]
            total_loss += return_dict["total_loss"]
            total_correct += return_dict["total_correct"]


        ave_loss = total_loss / num_samples
        acc = total_correct / num_samples

        return ave_loss, acc

    def base_evaluate_fold(self,nn, eval_on_train ,**kwargs):
        """
            Evaluate results on training data/test data.
        """
        num_samples = 0
        total_loss = 0
        total_correct = 0
        for c in self.clients:
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Evaluate locally
            if eval_on_train:
                return_dict = c.evaluate_train_fold(nn,**kwargs)
            else:
                return_dict = c.evaluate_test_fold(nn,**kwargs)

            num_samples += return_dict["num_samples"]
            total_loss += return_dict["total_loss"]
            total_correct += return_dict["total_correct"]


        ave_loss = total_loss / num_samples
        acc = total_correct / num_samples

        return ave_loss, acc