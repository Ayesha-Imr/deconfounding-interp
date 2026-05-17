"""Pipeline stage interfaces.

Concrete GPU/API backends should live here and consume manifest jobs. Keeping these
stages behind interfaces lets trait and model configs stay stable.
"""
