"""
Prompts dispatch by dataset: gender, rosbank, df_2024.
Import from config (must be loaded first) to get DATASET_NAME.
"""

from configs.config import DATASET_NAME

if DATASET_NAME == "gender":
    from prompts.gender import (
        create_system_prompt_rules,
        create_system_prompt,
        create_user_prompt,
        create_multi_user_prompt,
    )
elif DATASET_NAME == "rosbank":
    from prompts.rosbank import (
        create_system_prompt_rules,
        create_system_prompt,
        create_user_prompt,
        create_multi_user_prompt,
    )
else:
    # df_2024 or any other churn dataset
    from prompts.df_2024 import (
        create_system_prompt_rules,
        create_system_prompt,
        create_user_prompt,
        create_multi_user_prompt,
    )

__all__ = [
    "create_system_prompt_rules",
    "create_system_prompt",
    "create_user_prompt",
    "create_multi_user_prompt",
]
