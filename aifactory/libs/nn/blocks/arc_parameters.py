from dataclasses import dataclass
from typing import Union


@dataclass
class BasicBlockParameters:
    input_channel: int
    mid_channels: list | tuple | None
    output_channel: int
    activation: dict | None


def from_dict(cls, data):
    if data is None:
        return None
    # If data is already of the target type, return it directly
    if isinstance(data, cls):
        return data
    # Handle list: recursively convert each element
    if hasattr(cls, '__origin__') and cls.__origin__ is list:  # handling typing.List
        elem_type = cls.__args__[0]
        return [from_dict(elem_type, item) for item in data]
    # Handle Optional or Union
    if hasattr(cls, '__origin__') and cls.__origin__ is Union:
        # Simple handling: try the first non-None type (demonstration only)
        for typ in cls.__args__:
            if typ is not type(None):
                return from_dict(typ, data)
        return None
    # Handle dataclass
    if hasattr(cls, '__dataclass_fields__'):
        field_types = {f: cls.__dataclass_fields__[f].type for f in cls.__dataclass_fields__}
        # Recursively build for each field
        kwargs = {}
        for field, field_type in field_types.items():
            if field in data:
                kwargs[field] = from_dict(field_type, data[field])
            else:
                # If field has a default value, skip; otherwise may need to handle missing (simplified here)
                pass
        return cls(**kwargs)
    # Primitive types return directly
    return data

