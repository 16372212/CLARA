from typing import Dict
from util.mongo_util import get_mongo_client
from util.const import host, databases_name, less_20_families
import pickle

label_name = "label_num.pkl"


def get_labels_from_file(filename: str) -> Dict[str, Dict[str, str]]:
    f = open(filename, 'r')
    all_data = f.read().split('\n')
    labels: Dict[str, Dict[str, str]] = {}
    for data in all_data:
        if ',' in data:
            data = data.split(',')
            file_hash = data[0]
            type1 = data[1]
            type2 = data[2]
            labels[file_hash] = {}
            labels[file_hash]['label'] = type1
            labels[file_hash]['family'] = type2
    f.close()
    return labels


def get_labels_from_mongo(labels: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """从mongo中的db中读取到每个"""
    label_family_num: Dict[str, Dict[str, int]] = {}
    labels_from_mongo: Dict[str, Dict[str, str]] = {}
    for database_name in databases_name:
        print(f"getting label from database: {database_name}")
        client = get_mongo_client(host)
        collections = client[database_name]['analysis']
        file_collection = client[database_name]['report_id_to_file']
        # api_collection = client['db_calls'][dbcalls_dict[database_name]]
        cursor = collections.find(no_cursor_timeout=True)
        for x in cursor:
            # 进程list,包括样本
            # 获取hash
            rows = file_collection.find(filter={'_id': str(x['_id'])})
            for row in rows:
                file_hash = row['file_hash']
                if file_hash is None or file_hash not in labels:
                    continue
                big_label = labels[file_hash]['label']
                family = labels[file_hash]['family']
                labels_from_mongo[file_hash] = {}
                labels_from_mongo[file_hash]['label'] = big_label
                labels_from_mongo[file_hash]['family'] = family
                if big_label not in label_family_num:
                    label_family_num[big_label] = {}
                    label_family_num[big_label][family] = 1
                else:
                    if family in label_family_num[big_label]:
                        label_family_num[big_label][family] += 1
                    else:
                        label_family_num[big_label][family] = 1
        client.close()
    print(label_family_num)
    with open(label_name, "wb") as f1:
        pickle.dump(label_family_num, f1, pickle.HIGHEST_PROTOCOL)
    return labels_from_mongo


def get_20_family():
    """查看family个数，针对每个大类，选择差不多数量的数据，看是否能提升准确率"""
    with open(label_name, "rb") as f:
        label_family_num = pickle.load(f)
    print(label_family_num)
    # 有多少>20个数的家族，每个label这样的家族个数有多少
    label_count = 0
    labs = []
    total_family_count = 0
    for label in label_family_num:
        tmp_count = 0
        for fam in label_family_num[label]:

            if label_family_num[label][fam] < 20:
                continue
            total_family_count += 1
            tmp_count += 1
        if tmp_count != 0:
            labs.append(label)
            label_count += 1
            print(f'label: {label}, samples: {tmp_count}')
    print(f'一共{total_family_count}个family, {label_count}个label 包含的样本数量>20')  # 103, 12

    print(labs)


def find_kaggle_in_mongo():
    for database_name in databases_name:
        print(f"getting label from database: {database_name}")
        client = get_mongo_client(host)
        collections = client[database_name]['analysis']
        file_collection = client[database_name]['report_id_to_file']
        # api_collection = client['db_calls'][dbcalls_dict[database_name]]
        cursor = collections.find(no_cursor_timeout=True)
        for x in cursor:
            # 进程list,包括样本
            # 获取hash
            rows = file_collection.find(filter={'_id': str(x['_id'])})
            for row in rows:
                file_hash = row['file_hash']
                if file_hash is None:
                    continue
                if file_hash == "" or file_hash == "268a6e44780afd1605558fbba0b15ee6"\
                        or file_hash == "b04bced21f6feee3f02e418e1287fb4f" or file_hash == "8cb90da0cb1463638fefdc4150ece7b3":
                    print("找到了!!")
                    return


if __name__ == "__main__":
    # labels: Dict[str, Dict[str, str]] = get_labels_from_file("../label/sample_result.txt")
    # all_labels = get_labels_from_mongo(labels)
    get_20_family()
    # find_kaggle_in_mongo()