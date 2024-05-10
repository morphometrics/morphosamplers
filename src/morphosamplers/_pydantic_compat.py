"""This is a pydantic v2 compatibility approach that uses pydantic.v1
This implementation is using the work by Grzegorz Bokota @Czaki
in the napari codebase: https://github.com/napari/napari/pull/6358
"""
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
    # pydantic v1
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
