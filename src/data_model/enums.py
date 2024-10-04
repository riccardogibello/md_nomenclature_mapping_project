from enum import Enum
from typing import Any, Union


class AbstractEnum(Enum):

    def __to_string(
            self,
            enum_value__string__mapping: dict["AbstractEnum", str],
    ):
        for enum_value in enum_value__string__mapping.keys():
            if self == enum_value:
                return enum_value__string__mapping[enum_value]
        raise Exception('Enum not convertible: ' + str(self))

    @staticmethod
    def get_enum_from_value(
            value: Union[int, str],
            enum_class: type["AbstractEnum"],
    ) -> Any:
        for enum_value in enum_class:
            if value == enum_value.value:
                return enum_value
        raise Exception('Enum not recognized for ' + str(value) + " in " + str(enum_class))


class HighRiskMedicalDeviceType(AbstractEnum):
    NOT_SPECIFIED = 'not_specified'
    CARDIOVASCULAR = 'cardiovascular'
    DIABETIC = 'diabetic'
    ORTHOPEDIC = 'orthopedic'


class MatchType(AbstractEnum):
    EXACT = "exact"
    SIMILARITY = "similarity"
    MAPPING_ALGORITHM = "mapping_algorithm"
    NOT_DEFINED = "not_defined"


class CountryName(AbstractEnum):
    ITALY = 'Italy'
    PORTUGAL = 'Portugal'
    USA = 'USA'


class TranslationDirection(AbstractEnum):
    FROM_GMDN_TO_EMDN = 0
    FROM_EMDN_TO_GMDN = 1
