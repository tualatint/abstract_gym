import mysql.connector
from mysql.connector import Error
import random
import numpy as np
import copy
from torch.utils import data

import __init__
from abstract_gym.utils.db_data_type import Trial_data, SAR_point, Saqt_point


class NumpyMySQLConverter(mysql.connector.conversion.MySQLConverter):
    """ A mysql.connector Converter that handles Numpy types """

    def _float32_to_mysql(self, value):
        return float(value)

    def _float64_to_mysql(self, value):
        return float(value)

    def _int32_to_mysql(self, value):
        return int(value)

    def _int64_to_mysql(self, value):
        return int(value)


def speed_range_check(v):
    if v > np.pi:
        v -= np.pi * 2.0
    if v < -np.pi:
        v += np.pi * 2.0
    return v


def validate_data(r_list):
    if len(r_list) != 7:
        return False
    for r in r_list:
        if r is None:
            return False
    return True


class MySQLInterface:
    def __init__(self,
                 user='root',
                 password='123',
                 database='abstract_data',
                 trial_table_name='trial_table',
                 sar_table_name='data_points_table',
                 saqt_table_name='saqt_1'):
        self.conn = mysql.connector.connect(user=user, password=password, database=database)
        self.conn.set_converter_class(NumpyMySQLConverter)
        self.trial_table_name = trial_table_name
        self.sar_table_name = sar_table_name
        self.saqt_table_name = saqt_table_name

        self.cursor = self.conn.cursor(buffered=True)
        self.cursor.execute("show databases")
        for x in self.cursor:
            print(x)
        try:
            self.cursor.execute('use abstract_data')
            '''
            Id: the index of current trial
            Step: time step of current data point
            J0c: current joint 0 state
            '''
            self.cursor.execute(
                'create table if not exists ' + self.saqt_table_name + '(Id int NOT NULL, J0s float, J1s float, '
                                                                       'V0s float, V1s float, J0t float, J1t float, '
                                                                       'A0 float, A1 float, Qt float,PRIMARY KEY (Id)) ')
            self.cursor.execute(
                'create table if not exists ' + self.sar_table_name + '(Primary_id int NOT NULL AUTO_INCREMENT, '
                                                                      'Id int NOT NULL, Step int, J0c float, '
                                                                      'J1c float, Action0 float, Action1 float, '
                                                                      'Reward smallint,PRIMARY KEY (Primary_id))')
            self.cursor.execute(
                'create table if not exists ' + self.trial_table_name + '(Id int NOT NULL AUTO_INCREMENT, Succ '
                                                                        'boolean, Duration smallint,J0s float, '
                                                                        'J1s float,J0e float, J1e float,PRIMARY KEY ('
                                                                        'Id))')
            self.conn.commit()
        except Error as e:
            print("Something went wrong during sql initialization.", e)
            self.conn.rollback()
        finally:
            pass
        self.trial_max = self.get_current_trial_index()
        self.data_point_max = self.get_current_sar_primary_index()
        self.saqt_max = self.get_current_saqt_index()

    def delete_all_tables(self):
        try:
            delete_instruction = 'drop table ' + self.trial_table_name
            self.cursor.execute(delete_instruction)
            delete_instruction2 = 'drop table ' + self.sar_table_name
            self.cursor.execute(delete_instruction2)
            delete_instruction3 = 'drop table ' + self.saqt_table_name
            self.cursor.execute(delete_instruction3)
            self.conn.commit()
        except Error as e:
            print("Error during table deleting:", e)
            self.conn.rollback()

    def insert_trial(self, t):
        insert_instruction = 'insert into ' + self.trial_table_name + '(Succ, Duration, J0s, J1s, J0e, J1e) values (' \
                                                                      '%s, %s, %s, %s, %s, %s) '
        insert_data = (t.succ, t.duration, t.j0s, t.j1s, t.j0e, t.j1e)
        try:
            self.cursor.execute(insert_instruction, insert_data)
            self.conn.commit()
        except Error as e:
            print("Error during trial data insertion:", e)
            self.conn.rollback()

    def get_current_trial_index(self):
        return self.get_current_index(self.trial_table_name)

    def get_current_sar_primary_index(self):
        return self.get_current_index(self.sar_table_name, col='Primary_id')

    def get_current_saqt_index(self):
        return self.get_current_index(self.saqt_table_name)

    def get_current_index(self, table_name, col='Id'):
        self.cursor.execute('select max(' + col + ') from ' + table_name)
        index = int(self.cursor.fetchone()[0])
        self.conn.commit()
        if index is None:
            idx = 0
        else:
            idx = int(index)
        return idx

    # def get_random_trial_data(self):
    #     """
    #     randomly pick one trial data to train NN
    #     :return:
    #     """
    #     try:
    #         max_id = self.get_current_trial_index()
    #         result = None
    #         while result is None:
    #             rand_id = random.randint(1, max_id)
    #             result = self.get_trial_data_by_id(rand_id)
    #         return result
    #     except Error as e:
    #         print("Error getting trial data point:", e)

    def get_trial_data_by_id(self, id):
        try:
            if id == 0:
                id += 1
            if 1 <= id <= self.trial_max:
                instruction = 'select * from ' + self.trial_table_name + ' where Id = {} limit 1'.format(id)
                self.cursor.execute(instruction)
                self.conn.commit()
                result = self.cursor.fetchone()
                r_list = list(result)
                if validate_data(r_list):
                    return r_list[3:7], r_list[1:3]  # {J0s, J1s, J0e, J1e}, {Succ, Duration}
                else:
                    print("r list invalid.")
            else:
                print("Given id out of range.{}/{}".format(id, self.trial_max))
        except Error as e:
            print("Error getting trial data point:", e)

    def sample_sarst(self, batch_size):
        sarst_list = []
        while len(sarst_list) < batch_size:
            rand_idx = random.randint(1, self.data_point_max)
            #rand_idx = random.randint(1, np.int(1e7))
            sarst = self.get_sars_data_by_id(rand_idx, self.data_point_max)
            if sarst is None:
                continue
            sarst_list.append(sarst)
        return sarst_list

    def visit_sarst(self, batch_size, index):
        sarst_list = []
        idx = copy.deepcopy(index)
        while len(sarst_list) < batch_size:
            sarst = self.get_sars_data_by_id(idx, self.data_point_max)
            if sarst is None:
                continue
            idx += 1
            sarst_list.append(sarst)
        return sarst_list

    def sample_saqt(self, batch_size):
        saqt_list = []
        while len(saqt_list) < batch_size:
            rand_idx = random.randint(1, self.saqt_max)
            saqt = self.get_saqt_by_id(rand_idx)
            if saqt is None:
                continue
            saqt_list.append(saqt)
        return saqt_list

    def get_saqt_by_id(self, idx):
        try:
            if not (0 <= idx <= self.saqt_max):
                print("saqt id out of range")
                return None
            if idx == 0:
                idx = 1
            instruction = 'select * from ' + self.saqt_table_name + ' where Id = {} limit 1'.format(idx)
            self.cursor.execute(instruction)
            result = self.cursor.fetchall()
            self.conn.commit()
            if len(result) != 1:
                print("Error reading saqt with index :", idx)
                return None
            # saqt = (Id | J0s | J1s | V0s | V1s | J0t | J1t | A0 | A1 | Qt)
            saqt = list(result[0])
            sa = saqt[1:9]
            qt = saqt[9]
            return sa, qt
        except Error as e:
            print("Error getting saqt data point:", e)

    def get_sars_data_by_id(self, primary_id, max_pri_id):
        try:
            if not (0 <= primary_id <= max_pri_id):
                print("primary_id out of range")
                return None
            instruction = 'select * from ' + self.sar_table_name + ' where Primary_id = {} or Primary_id = {} or ' \
                                                                   'Primary_id = {} limit 3'.format(
                primary_id, primary_id + 1, primary_id + 2)
            self.cursor.execute(instruction)
            result = self.cursor.fetchall()
            self.conn.commit()
            if len(result) < 3:
                print("Error during reading data point table.")
                return None
            l1 = list(result[0])
            l2 = list(result[1])
            l3 = list(result[2])
            trial_id = l1[1]
            trial_data = self.get_trial_data_by_id(trial_id)
            if trial_data is None:
                print("Did not find the corresponding trial record.")
                return None
            trial_list = list(trial_data)
            if l1[1] == l2[1] and l1[2] == (l2[2] - 1):
                """
                if they have the same trial id and step = next_step -1
                l1 = (primary_id, id, step, j1, j2, a1, a2, r)
                """
                v11 = speed_range_check(l2[3] - l1[3])
                v12 = speed_range_check(l2[4] - l1[4])
                if l1[1] == l3[1] and l1[2] == (l3[2] - 2):
                    v21 = speed_range_check(l3[3] - l2[3])
                    v22 = speed_range_check(l3[3] - l2[3])
                else:
                    v21 = 0.0
                    v22 = 0.0
                """
                sarst = (j1c, j2c, a1, a2, r, v11, v12, j1n, j2n, v21, v22, j1e, j2e, primary_id)
                """
                sarst = l1[3:8] + [v11, v12] + l2[3:5] + [v21, v22] + trial_list[0][2:4] + [l1[0]]
            else:
                """
                if it's the last one of the episode:
                """
                sarst = l1[3:8] + [0.0, 0.0] + l1[3:5] + [0.0, 0.0] + trial_list[0][2:4] + [l1[0]]
            return sarst
        except Error as e:
            print("Error getting sars data point:", e)

    def insert_sar_data_point(self, p):
        insert_instruction = 'insert into ' + self.sar_table_name + '(Id, Step, J0c, J1c, Action0, Action1, Reward) ' \
                                                                    'values (%s, %s, %s, %s, %s, %s, %s) '
        insert_data = (p.trial_id, p.step, p.j0c, p.j1c, p.a0, p.a1, p.reward)
        try:
            self.cursor.execute(insert_instruction, insert_data)
            self.conn.commit()
        except Error as e:
            print("Error during sar data point insertion:", e)
            self.conn.rollback()

    def insert_labeled_saqt_data(self, saqt_list):
        try:
            insert_data_list = []
            insert_instruction = 'insert into ' + self.saqt_table_name + '(Id, J0s, J1s, V0s, V1s, J0t, J1t, A0, A1,Qt) ' \
                                                                         'values (%s, %s, %s, %s, %s, %s, %s, ' \
                                                                         '%s, %s, %s) ON DUPLICATE KEY UPDATE Qt = VALUES(Qt)'

            for saqt in saqt_list:
                insert_data = (
                saqt.id, saqt.j0s, saqt.j1s, saqt.v0s, saqt.v1s, saqt.j0t, saqt.j1t, saqt.a0, saqt.a1, saqt.qt)
                insert_data_list.append(insert_data)
            self.cursor.executemany(insert_instruction, insert_data_list)
            self.conn.commit()
        except Error as e:
            print("Error during saqt data point insertion:", e)
            self.conn.rollback()

    def shutdown(self):
        self.cursor.close()
        self.conn.close()


class TrialDataSampler(data.Dataset):
    def __init__(self, db):
        self.db = db
        self.max_idx = self.__len__()

    def __len__(self):
        return self.db.get_current_trial_index()

    def __getitem__(self, idx):
        return self.db.get_trial_data_by_id(idx)


class DataPointsSampler(data.Dataset):
    def __init__(self, db):
        self.db = db
        self.max_pri_idx = self.__len__()

    def update_len(self):
        self.max_pri_idx = self.__len__()

    def __len__(self):
        return self.db.get_current_sar_primary_index()

    def __getitem__(self, idx):
        return self.db.get_sars_data_by_id(idx, self.max_pri_idx)

class SaqtDataSampler(data.Dataset):
    def __init__(self, db):
        self.db = db
        self.max_idx = self.__len__()

    def __len__(self):
        return np.int64(1.74e7)
        #return self.db.saqt_max

    def __getitem__(self, idx):
        return self.db.get_saqt_by_id(idx)

if __name__ == "__main__":
    db = MySQLInterface()
    #dummy_trial = Trial_data(True, 35, 0.013, 0.124, 0.331, 0.425)
    #dummy_trial_2 = Trial_data(False, 34, 0.011, 0.224, 0.231, 0.125)
    # db.insert_trial(dummy_trial)
    # db.insert_trial(dummy_trial_2)
    #idx = db.get_current_trial_index()
    #print("current index :", idx)
    #dummy_data_point = SAR_point(idx, 0, 0.1, 0.2, 0.01, -0.03, 0)
    # db.insert_sar_data_point(dummy_data_point)
    # db.get_random_trial_data()
    # db.get_random_sars_data_point()
    # sarst = db.get_sars_data_by_id(6400281, db.get_current_sar_primary_index())
    # print("sar2", sarst)
    # max_id = 6400281
    # saqt = Saqt_point(6926263, 0.15,0.5,0.01,0.02,0.56,0.41,0.001,0.002, 0.6)
    # saqt_1 = Saqt_point(4401873, 0.15,0.5,0.01,0.02,0.56,0.41,0.001,0.002, 0.22)
    # saqt_2 = Saqt_point(6926262, 0.15,0.5,0.01,0.02,0.56,0.41,0.001,0.002, 0.64)
    # saqt_3 = Saqt_point(4401872, 0.15,0.5,0.01,0.02,0.56,0.41,0.001,0.002, 0.25)
    # saqt_list = [saqt, saqt_1]
    # saqt_list2 = [saqt_2, saqt_3]
    # for j in range(10):
    #     for i in range (10):
    #         rand_id = random.randint(1, max_id)
    #         rand_qt = random.random()
    #         saqt = Saqt_point(rand_id, 0.15, 0.5, 0.01, 0.02, 0.56, 0.41, 0.001, 0.002, rand_qt)
    #         saqt_list.append(saqt)
    #     db.insert_labeled_saqt_data(saqt_list)
    # db.insert_labeled_saqt_data(saqt_list2)
    li = db.get_saqt_by_id(5)
    print(li)
    db.shutdown()
