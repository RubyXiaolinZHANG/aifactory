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


if __name__ == "__main__":
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