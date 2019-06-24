from typing import Any, Optional

class Label(object):
    _value: Optional[str]

    def __init__(self, value: Optional[str] = None) -> None:
        self._value = value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Label):
            return self._value == other._value
        else:
            return False

    def __repr__(self) -> str:
        return 'Label(%r)' % self._value

    def __str__(self) -> str:
        return self._value

    def set_label(self, label_value: str) -> None:
        if self.assigned():
            raise ValueError('Label already has value: %s' % self._value)
        self._value = label_value

    def assigned(self) -> bool:
        return self._value is not None


