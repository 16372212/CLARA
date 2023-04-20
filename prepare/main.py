from draw_graph import create_graph_matrix
from genApiMatrix import readApiFromMongo
from Sample import Node, Sample


if __name__ == "__main__":
    """
    0. create api matrix attribute, adjust api attribute length
    """
    # readApiFromMongo()

    """ 
    1. create graph matrix. Read data from mongoDB, use some of them to create
        output: graph.pkl, sample_list.pkl
    """
    create_graph_matrix()

    """
    2. 创建dgl，这里并为保存
    """
    from matrix_to_huge_dgl import draw_aug_dgls, draw_dgl_from_matrix
    huge_graph, sample_id_lists, family_label_lists, big_label_lists = draw_dgl_from_matrix()

    """
    3. Data augmentation. creating augmented graphs
        output: gcc_input/aug_graphs_15/  , subgraphs_train_data.bin
    """
    draw_aug_dgls(huge_graph, sample_id_lists, family_label_lists, big_label_lists)
