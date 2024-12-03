from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import time
import torch

# print("is enable?",torch.torch_version.__version__)
def train_client(args):
    client, kwargs = args
    return client.local_train(**kwargs)



class FDUTrainer(BaseTrainer):
    #len(self.update_table) =N
    current_round=-16

    def record(self):
        record=[]
        for i in range(250):#
            record.append(0)
        return record

    def counting(self):
        counting = []
        for i in range(len(self.update_table)):

            counting.append(0)
        return counting


    def __init__(self, options, dataset, result_dir='results'):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])

        self.update_table = []  # store previous updates

        self.previous_update_table=[]# 保存 VARP公式（4）的y（t）

        super(FDUTrainer, self).__init__(options, dataset, model, self.optimizer, result_dir)

    def set_0_update_table(self):
        solns, _ = self.local_train(0, self.clients)
        print("this is from set_0_update_table")
        self.update_table =[torch.zeros_like(solns[i][1]) for i in
                            range(len(self.clients))]
        self.previous_update_table=[torch.zeros_like(solns[i][1]) for i in
                            range(len(self.clients))]


    def initialize_update_table_varp(self):
        solns, _ = self.local_train(0, self.clients)
        self.update_table = [(1 / self.optimizer.get_current_lr()) * (solns[i][1] - self.latest_model) for i in
                            range(len(self.clients))]
        # self.update_table = [(1 / 0.0001) * (solns[i][1] - self.latest_model) for i in
        #                      range(len(self.clients))]

    def initialize_update_table(self):
        solns, _ = self.local_train(0, self.clients)
        self.update_table = [(1 / self.optimizer.get_current_lr()) * (solns[i][1] - self.latest_model) for i in
                            range(len(self.clients))]
        # self.update_table = [(1 / 0.0001) * (solns[i][1] - self.latest_model) for i in
        #                      range(len(self.clients))]

    def train(self):
        print(f'>>> Select {self.clients_per_round} clients per round \n')
        self.latest_model = self.worker.get_flat_model_params().detach()
        self.denomi_list = []
        table_initialized = True
        self.set_0_update_table()
        truth_round = 0
        self.record1 = self.record()

        for round_i in range(self.num_round):
            start_time = time.time()
            self.current_round += 1

            self.selected_clients = self.get_avail_clients(seed=round_i)

            if table_initialized:
                # 准备参数字典传递给每个客户端的 local_train
                client_args = [(client, {'model': self.latest_model, 'round_number': truth_round}) for client in
                               self.selected_clients]

                # 并行执行本地训练
                with ProcessPoolExecutor() as executor:
                    results = list(executor.map(train_client, client_args))

                # 解包结果
                solns = [result[0] for result in results]
                stats = [result[1] for result in results]

                print(truth_round, " this is the truth_round")
                for i in range(len(self.update_table)):
                    self.previous_update_table[i] = self.update_table[i]

                print(
                    "#######################################################################################################################################")
                for idx, c in enumerate(self.selected_clients):
                    self.record1[c.cid] = round_i - 15
                    self.update_table[c.cid] = (
                                1 / self.optimizer.get_current_lr() * (solns[idx][1] - self.latest_model))

                print(
                    "#######################################################################################################################################")

                self.aggregate2()

                self.optimizer.inverse_prop_decay_learning_rate(truth_round + 1)

                train_loss, train_acc = self.evaluate_train()
                test_loss, test_acc = self.evaluate_test()
                out_dict = {'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss,
                            'test_acc': test_acc}
                print("training loss & acc", train_loss, train_acc)
                print("test loss & acc", test_loss, test_acc)
                self.logger.log(round_i, out_dict)
                self.logger.dump()
                truth_round += 1

                end_time = time.time()  # 结束计时
                elapsed_time = end_time - start_time
                print(f"Total runtime: {elapsed_time:.2f} seconds")

        denominator_ndarray = np.array(self.denomi_list)
        #np.savetxt('denominator6.txt', denominator_ndarray)
    def train_last(self,model_name,algo,c_number,dir):
        print("this is last last round")
        self.local_train_last_round(self.num_round, self.clients, model_name,algo,c_number,dir)
    def test_last(self, model_name, algo, c_number, dir):
        print("this is last last round")
        self.local_test_last_round(self.num_round, self.clients, model_name, algo, c_number, dir)
    def weight(self):
        def aggregate(self):
            sum_w = 0
            s = len(self.selected_clients)
            print(s, " this is the number of selected_clients in aggregate define")

            psi_i = 0
            denominator = 0
            sum_psi = 0
            sum_tau = 0
            for j in range(len(self.update_table)):
                tau_j = self.current_round - self.record1[j]
                # print(tau_j,"tau_j",self.current_round,self.record1[j])
                denominator = pow((tau_j + 1), 0.37) + denominator  #
                # denominator = pow(1.02, (tau_j )) + denominator

            print(denominator, "sum of tau+1 ")

            for i in range(len(self.update_table)):
                tau_i = self.current_round - self.record1[i]
                if tau_i > 0:

                    numerator = pow((tau_i + 1), 0.37) * len(self.update_table) * 1.13  #
                    # numerator = (pow(1.02, tau_i) * len(self.update_table))
                    psi_i = numerator / denominator
                else:
                    psi_i = 1

                sum_psi = sum_psi + psi_i

                # sum_w = sum_w + self.update_table[i] * pow(1.098,(self.current_round-self.record1[i]))
                sum_w = sum_w + self.update_table[i] * psi_i

            # self.latest_model = self.latest_model +self.optimizer.get_current_lr()* sum_w/ len(self.clients)*1.13
            self.latest_model = self.latest_model + self.optimizer.get_current_lr() * sum_w / len(self.clients) * 1.13

        self.num_round


    def aggregate2(self):

        s = len(self.selected_clients)  # 被选取（可用）clinet的个数
        print(s," this is the number of selected_clients")
        first_term=0  #VARP 公式4的第一个term
        # limit_in_5=0
        for (idx, c) in enumerate(self.selected_clients):
         # if limit_in_5<6:
            first_term=first_term+(self.update_table[c.cid]-self.previous_update_table[c.cid])#计算VARP 公式4的第一个term
            # limit_in_5=limit_in_5+1
        v=first_term/s
        second_term = 0  # VARP 公式4的第二个term
        for j in range(len(self.previous_update_table)):
            second_term=second_term+self.previous_update_table[j]

        v=v+second_term/len(self.previous_update_table)




        self.latest_model = self.latest_model +self.optimizer.get_current_lr()*v
        # self.latest_model = self.latest_model +  0.0001* v
