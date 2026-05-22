"""Tests for ``CatalogElementMixin``.

Vendored from autonlab/ngautonml@main (Apache-2.0).
"""

# Copyright (c) 2023 Carnegie Mellon University
# Vendored under Apache-2.0; see catalog.py.

import pytest

from .memory_catalog import CatalogNameError
from .catalog_element_mixin import CatalogElementMixin


class Widget(CatalogElementMixin):
    """Plain mixin user."""


class WidgetNameAndTags(Widget):
    _name = 'widget_with_name_and_tags'
    _tags = {
        'widget_tag': ['widget_value'],
    }


class WidgetUppercase(Widget):
    _name = 'Widget_with_Uppercase'
    _tags = {
        'Widget_Tag': ['Widget_Value'],
    }


class WidgetNameNoTags(Widget):
    _name = 'widget_with_name'


def test_sunny_day():
    dut = WidgetNameAndTags()
    assert dut.name == 'widget_with_name_and_tags'
    assert dut.tags['widget_tag'] == ['widget_value']


def test_widget_name_no_tags():
    dut = WidgetNameNoTags()
    assert dut.name == 'widget_with_name'
    assert not dut.tags


def test_missing_name_should_fail():
    with pytest.raises(CatalogNameError, match='Widget'):
        Widget()


def test_uppercase():
    """Confirm that name and tags are all forced to lower case."""
    dut = WidgetUppercase()
    assert dut.name == 'widget_with_uppercase'
    assert dut.tags['widget_tag'] == ['widget_value']
