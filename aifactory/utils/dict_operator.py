import types


def dict_to_struct_recursive(data_dict):
    """Recursively converts a dictionary (including nested dictionaries) into a SimpleNamespace structure."""
    if not isinstance(data_dict, dict):
        return data_dict  # Base types are returned directly

    # Process nested dictionaries: recursive conversion
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = dict_to_struct_recursive(value)
        elif isinstance(value, list):
            # Optionally handles dictionaries within the list
            data_dict[key] = [dict_to_struct_recursive(item)
                              if isinstance(item, dict) else item
                              for item in value]

    return types.SimpleNamespace(**data_dict)



def remove_key_recursive(obj, key):
    """
    递归删除对象中所有键名为 key 的键值对。
    支持的数据结构：字典、列表、元组、集合（集合内元素必须可哈希）。
    对于其他类型直接返回。
    """
    if isinstance(obj, dict):
        # 处理字典：跳过要删除的键，递归处理值
        return {
            k: remove_key_recursive(v, key)
            for k, v in obj.items()
            if k != key
        }
    elif isinstance(obj, list):
        # 处理列表：递归处理每个元素
        return [remove_key_recursive(item, key) for item in obj]
    elif isinstance(obj, tuple):
        # 处理元组：递归处理每个元素后重建元组
        return tuple(remove_key_recursive(item, key) for item in obj)
    elif isinstance(obj, set):
        # 处理集合：递归处理每个元素后重建集合（元素必须可哈希）
        return {remove_key_recursive(item, key) for item in obj}
    else:
        # 其他类型（int, str, float, None, 自定义对象等）直接返回
        return obj


########################################################################################################################
# the follows are test cases

def test_dict_to_struct_recursive():
    company_data = {
        'name': 'TechCorp',
        'ceo': {'name': 'Bob', 'age': 45},
        'departments': [
            {'name': 'R&D', 'size': 50},
            {'name': 'Sales', 'size': 30}
        ]
    }
    company = dict_to_struct_recursive(company_data)

    print(f"公司: {company.name}")
    print(f"CEO: {company.ceo.name}, 年龄: {company.ceo.age}")
    print(f"第一个部门: {company.departments[0].name}, 规模: {company.departments[0].size}")


def test_remove_key_recursive():
    # 示例
    data = {
        "a": "a",
        "func": "func1",
        "b": [{"func": "func2"}, {"func": "func3"}],
        "c": {"func": "func4"}
    }

    result = remove_key_recursive(data, "func")
    print(result)

if __name__ == "__main__":
    test_remove_key_recursive()