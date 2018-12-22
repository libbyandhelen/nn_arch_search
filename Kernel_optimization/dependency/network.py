from collections import Counter


class Network:
    def __init__(self, architecture, level, label_compressor, B):
        # print(type(architecture))
        self.node_label = []
        self.connection_list = [[] for i in range(2 * B * level)]
        self.B = B
        self.arch = architecture
        # self.op1 = architecture["op1"]
        # self.op2 = architecture["op2"]
        # self.op1_prev = int(architecture["op1_prev"])
        # self.op2_prev = int(architecture["op2_prev"])
        self.level = level
        self.label_compressor = label_compressor

        self.construct_node_label()
        self.construct_connection_list()
        # print(self.connection_list)

    def construct_node_label(self):
        for i in range(self.level):
            if i in [self.level // 3, 2 * self.level // 3]:
                for op in self.arch["reduce"]:
                    self.node_label.append(self.label_compressor.compress(op[0]))
            else:
                for op in self.arch["normal"]:
                    self.node_label.append(self.label_compressor.compress(op[0]))
                    # for i in range(2 * self.level):
                    #     if i % 2 == 0:
                    #         label = self.op1
                    #     else:
                    #         label = self.op2
                    #     self.node_label.append(self.label_compressor.compress(label))

    def mapping(self, x):
        return 2 * (x - 2)

    def construct_connection_list(self):
        node_counter = 0
        for i in range(self.level):
            if i in [self.level // 3, 2 * self.level // 3]:
                cell = self.arch["reduce"]
            else:
                cell = self.arch["normal"]
            for op in cell:
                # prev-prev level and prev cell
                if op[1] == 0 or op[1] == 1:
                    if op[1] == 0:
                        l = i - 2
                    else:
                        l = i - 1
                    if l < 0:
                        node_counter += 1
                        continue
                    if l in [self.level // 3, 2 * self.level // 3]:
                        concat = self.arch["reduce_concat"]
                    else:
                        concat = self.arch["normal_concat"]
                    for node in concat:
                        self.connection_list[l * 2*self.B + self.mapping(node)].append(node_counter)
                        self.connection_list[l * 2*self.B + self.mapping(node) + 1].append(node_counter)
                else:
                    self.connection_list[i * 2*self.B + self.mapping(op[1])].append(node_counter)
                    self.connection_list[i * 2*self.B + self.mapping(op[1]) + 1].append(node_counter)
                node_counter += 1
                # print("c", node_counter)

        # for i in range(1, self.level):
        #     if i - self.op1_prev >= 0:
        #         level = i - self.op1_prev
        #     else:
        #         level = i - 1
        #     self.connection_list[2 * level].append(2 * i)
        #     self.connection_list[2 * level + 1].append(2 * i)
        #
        #     if i - self.op2_prev >= 0:
        #         level = i - self.op2_prev
        #     else:
        #         level = i - 1
        #     self.connection_list[2 * level].append(2 * i + 1)
        #     self.connection_list[2 * level + 1].append(2 * i + 1)

    def single_multiset_label(self, root):
        root_label = str(self.node_label[root])
        child_labels = []
        for child in self.connection_list[root]:
            child_labels.append(self.node_label[child])
        child_labels.sort()
        if len(child_labels) == 0:
            return root_label

        return root_label + "*" + ','.join(map(str, child_labels))

    def run_iteration(self):
        label_multiset = []
        for i in range(len(self.node_label)):
            uncompress_label = self.single_multiset_label(i)
            # print(uncompress_label)
            label_multiset.append(self.label_compressor.compress(uncompress_label))

        # relabeling
        self.node_label = label_multiset

    def cal_graph_vector(self, label_set):
        counter = Counter(self.node_label)
        vector = []
        for label in label_set:
            vector.append(counter[label])
        return vector

