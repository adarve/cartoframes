"""Auth namespace contains the necessary tools to manage authentication by allowing
the user to set its CARTO credentials."""
from __future__ import absolute_import

from .credentials import Credentials, set_default_credentials, get_default_credentials

_default_credentials = None

__all__ = [
    'Credentials',
    'set_default_credentials',
    'get_default_credentials'
]
