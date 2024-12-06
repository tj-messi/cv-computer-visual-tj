import math
from collections import Counter, defaultdict

MAX_DEPTH = 4
depth = 0


class TreeNode:
    def __init__(self, attribute=None, classification=None):
        self.attribute = attribute  # 节点的属性
        self.branches = {}  # 子节点（键为分裂的值）
        self.classification = classification  # 叶子节点的分类结果

    def is_leaf(self):
        return self.classification is not None


class Sample:
    def __init__(self, attributes, classification):
        self.attributes = attributes  # 属性集合
        self.classification = classification  # 分类结果


def calculate_entropy(samples):
    counts = Counter(sample.classification for sample in samples)
    total = len(samples)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return entropy


def calculate_gain(samples, attribute_index):
    total_entropy = calculate_entropy(samples)
    split_samples = defaultdict(list)

    for sample in samples:
        split_samples[sample.attributes[attribute_index]].append(sample)

    split_entropy = sum(
        (len(subset) / len(samples)) * calculate_entropy(subset)
        for subset in split_samples.values()
    )
    return total_entropy - split_entropy


def get_majority_class(samples):
    if not samples:
        return "未知"  # 或者返回一个合理的默认值
    counts = Counter(sample.classification for sample in samples)
    return counts.most_common(1)[0][0]



def choose_best_attribute(samples, attributes):
    best_gain = -1
    best_index = -1
    for i, attribute in enumerate(attributes):
        gain = calculate_gain(samples, i)
        if gain > best_gain:
            best_gain = gain
            best_index = i
    return best_index


def stop_criteria(samples, attributes):
    global depth
    return len(attributes) == 0 or all(
        sample.classification == samples[0].classification for sample in samples
    ) or depth >= MAX_DEPTH


def build_decision_tree(samples, attributes):
    global depth
    if stop_criteria(samples, attributes):
        return TreeNode(classification=get_majority_class(samples))

    best_index = choose_best_attribute(samples, attributes)
    best_attribute = attributes[best_index]
    node = TreeNode(attribute=best_attribute)

    split_samples = defaultdict(list)
    for sample in samples:
        split_samples[sample.attributes[best_index]].append(sample)

    remaining_attributes = attributes[:best_index] + attributes[best_index + 1:]
    depth += 1

    for value, subset in split_samples.items():
        node.branches[value] = build_decision_tree(subset, remaining_attributes)

    depth -= 1
    return node


def predict(tree, attribute_values, attribute_names):
    if tree.is_leaf():
        return tree.classification

    # 找到当前属性在列表中的索引
    attribute_index = attribute_names.index(tree.attribute)
    attribute_value = attribute_values[attribute_index]

    # 如果在分支中找不到该值，则返回树中多数类作为默认预测
    if attribute_value not in tree.branches:
        # 返回节点下所有子节点的多数分类
        child_classes = [
            child.classification
            for child in tree.branches.values()
            if child.is_leaf()
        ]
        return get_majority_class([Sample([], cls) for cls in child_classes])

    # 递归预测
    return predict(tree.branches[attribute_value], attribute_values, attribute_names)


def calculate_accuracy(tree, samples, attribute_names):
    """
    计算样本集上的准确率
    :param tree: 决策树
    :param samples: 样本集
    :param attribute_names: 属性名称列表
    """
    correct = sum(
        1 for sample in samples
        if predict(tree, sample.attributes, attribute_names) == sample.classification
    )
    return correct / len(samples)


def main():
    training_data = [
        Sample(["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"], "是"),
        Sample(["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑"], "是"),
        Sample(["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"], "是"),
        Sample(["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘"], "是"),
        Sample(["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘"], "是"),
        Sample(["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘"], "否"),
        Sample(["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑"], "否"),
        Sample(["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘"], "否"),
        Sample(["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑"], "否"),
        Sample(["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑"], "否"),
    ]
    test_data = [
        Sample(["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑"], "是"),
        Sample(["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"], "是"),
        Sample(["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑"], "是"),
        Sample(["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑"], "是"),
        Sample(["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑"], "否"),
        Sample(["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘"], "否"),
        Sample(["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑"], "否"),
    ]
    attributes = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]

    decision_tree = build_decision_tree(training_data, attributes)
    train_accuracy = calculate_accuracy(decision_tree, training_data, attributes)
    test_accuracy = calculate_accuracy(decision_tree, test_data, attributes)

    print(f"训练集准确度: {train_accuracy}")
    print(f"测试集准确度: {test_accuracy}")



if __name__ == "__main__":
    main()
