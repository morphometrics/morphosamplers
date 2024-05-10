try:
    # pydantic v2
    from pydantic.v1 import (
        BaseModel,
        PrivateAttr,
        ValidationError,
        conint,
        root_validator,
        validator,
    )
    from pydantic.v1.generics import GenericModel
except ImportError:
    # pydantic v2
    from pydantic import (
        BaseModel,
        PrivateAttr,
        ValidationError,
        conint,
        root_validator,
        validator,
    )
    from pydantic.generics import GenericModel


__all__ = (
    "BaseModel",
    "ValidationError",
    "conint",
    "root_validator",
    "validator",
    "PrivateAttr",
    "GenericModel",
)
