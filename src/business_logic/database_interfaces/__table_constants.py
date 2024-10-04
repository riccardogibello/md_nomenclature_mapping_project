from src.data_model.enums import AbstractEnum


class TableColumnName(AbstractEnum):
    IDENTIFIER = 'identifier'
    NAME = 'name'
    CLEAN_NAME = 'clean_name'

    IS_AMERICAN_STATE = 'is_american_state'

    CLEAN_MANUFACTURER_ID = 'clean_manufacturer_id'
    STANDARDIZED_MANUFACTURER_ID = 'standardized_manufacturer_id'
    ORIGINAL_STATE_ID = 'original_state_id'

    CLEAN_DEVICE_ID = 'clean_device_id'
    STANDARDIZED_DEVICE_ID = 'standardized_device_id'
    DEVICE_NAME = 'device_name'
    DEVICE_TYPE = 'device_type'
    DEVICE_CLASS = 'device_class'
    HIGH_RISK_DEVICE_TYPE = 'high_risk_device_type'
    IS_IMPLANT = 'is_implant'
    IS_LIFE_SUSTAINING = 'is_life_sustaining'
    CATALOGUE_CODE = 'catalogue_code'
    MATCH_TYPE = 'match_type'
    SIMILARITY = 'similarity_value'
    MATCHED_NAME = 'matched_device_name'

    PRODUCT_CODE = 'product_code'
    PANEL = 'panel'
    x510K = 'x510k'
    MEDICAL_SPECIALTY = 'medical_specialty'
    SUBMISSION_TYPE_ID = 'submission_type_id'
    PHYSICAL_STATE = 'physical_state'
    TECHNICAL_METHOD = 'technical_method'
    TARGET_AREA = 'target_area'
    EMDN_ID = 'emdn_id'
    CODE = 'code'
    DESCRIPTION = 'description'
    IS_LEAF = 'is_leaf'
    GMDN_ID = 'gmdn_id'
    TERM_NAME = 'term_name'
    DEFINITION = 'definition'

    MAPPING_ID = 'mapping_id'
    FIRST_DEVICE_ID = 'first_device_id'
    SECOND_DEVICE_ID = 'second_device_id'
    COMPANY_NAME_SIMILARITY = 'company_name_similarity'
    STATUS = 'status'

    USER_ID = 'user_id'
    MOTIVATION = 'motivation'
    EMAIL = 'email'
    PASSWORD = 'password'
    USER_TYPE = 'user_type'
    TOKEN = 'token'

    EMBEDDING = 'embedding'
