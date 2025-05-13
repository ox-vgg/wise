import enum

class ContainsEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        if type(item) == cls:
            return enum.EnumMeta.__contains__(cls, item)
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseStrEnum(str, enum.Enum, metaclass=ContainsEnumMeta):
    pass

class IndexType(BaseStrEnum):
    IndexFlatIP = "IndexFlatIP"
    IndexIVFFlat = "IndexIVFFlat"
    IndexIVFPQ = "IndexIVFPQ"
