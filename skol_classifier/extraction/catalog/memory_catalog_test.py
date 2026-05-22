"""Tests for ``MemoryCatalog``.

Vendored from autonlab/ngautonml@main (Apache-2.0); skol-local import
paths and identical test bodies.  Run via:

    python -m pytest skol_classifier/extraction/catalog/memory_catalog_test.py -v
"""

# Copyright (c) 2023 Carnegie Mellon University
# Vendored under Apache-2.0; see catalog.py.

from pathlib import Path

import pytest

from .memory_catalog import CatalogDuplicateError
from .catalog_element_mixin import CatalogElementMixin
from .widgets.impl.widget_catalog import NamedWidget, WidgetCatalog, Widget


def test_register() -> None:
    """Test register with an implicit name."""
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    assert dut.register(test_widget) == 'testwidget'


def test_register_with_name() -> None:
    """Test register, specifying a name."""
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    assert dut.register(test_widget, 'TestName') == 'testname'


def test_register_duplicate() -> None:
    """Test duplicate registration."""
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    dut.register(test_widget, 'TestName')
    with pytest.raises(CatalogDuplicateError):
        dut.register(test_widget, 'TestName')


def test_lookup_by_name() -> None:
    """Sunny day test of lookup_by_name()."""
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    dut.register(test_widget, 'TestName')
    assert dut.lookup_by_name('TestName') == test_widget


def test_lookup_one_by_tag() -> None:
    """Find a list of one widget by arbitrary tag."""
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    dut.register(test_widget, 'TestWidget', tags={'userdef': ['TestTag']})
    assert len(dut.lookup_by_tag_and(userdef='TestTag')) == 1


def test_lookup_one_by_tag_case_difference() -> None:
    """Find a list of one widget by arbitrary tag."""
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    dut.register(test_widget, 'TestWidget', tags={'userdef': ['TestTag']})
    assert len(dut.lookup_by_tag_and(userdef='testtag')) == 1


def test_lookup_one_by_tag_case_difference_type() -> None:
    """Find a list of one widget by arbitrary tag."""
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    dut.register(test_widget, 'TestWidget', tags={'UserDef': ['TestTag']})
    assert len(dut.lookup_by_tag_and(userdef='TestTag')) == 1


def test_lookup_multi_by_tag() -> None:
    """Find a list of several widgets by arbitrary tag."""
    dut = WidgetCatalog()
    test_widgetone = NamedWidget('TestWidget')
    dut.register(test_widgetone, 'TestWidget', tags={'userdef': ['TestTag']})
    test_widgettwo = NamedWidget('TestWidgetTwo')
    dut.register(test_widgettwo, 'TestWidgetTwo', tags={'userdef': ['TestTag']})
    test_widgetthree = NamedWidget('TestWidgetThree')
    dut.register(test_widgetthree, 'TestWidgetThree', tags={'userdef': ['TestTag']})
    test_widgetfour = NamedWidget('TestWidgetFour')
    dut.register(test_widgetfour, 'TestWidgetFour', tags={'userdef': ['AnotherValue']})

    assert len(dut.lookup_by_tag_and(userdef='TestTag')) == 3


def test_lookup_multi_by_tag_type_diff() -> None:
    """Find a list of several widgets by arbitrary tag."""
    dut = WidgetCatalog()
    test_widgetone = NamedWidget('TestWidget')
    dut.register(test_widgetone, 'TestWidget', tags={'userdef': ['TestTag']})
    test_widgettwo = NamedWidget('TestWidgetTwo')
    dut.register(test_widgettwo, 'TestWidgetTwo', tags={'userdef': ['TestTag']})
    test_widgetthree = NamedWidget('TestWidgetThree')
    dut.register(test_widgetthree, 'TestWidgetThree', tags={'userdef': ['TestTag']})
    test_widgetfour = NamedWidget('TestWidgetFour')
    dut.register(test_widgetfour, 'TestWidgetFour', tags={'anothertype': ['TestTag']})

    assert len(dut.lookup_by_tag_and(userdef='TestTag')) == 3


def test_lookup_by_tag_not_found() -> None:
    """Fail to find by arbitrary tag."""
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    dut.register(test_widget, 'TestWidget')
    assert len(dut.lookup_by_tag_and(userdef='TestTag')) == 0


def test_lookup_by_tagtype_not_found() -> None:
    """Fail to find by arbitrary tag."""
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    dut.register(test_widget, 'TestWidget', tags={'userdef': ['TestTag']})
    assert len(dut.lookup_by_tag_and(anothertype='TestTag')) == 0


def test_get_tagtypes() -> None:
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    test_widget2 = NamedWidget('TestWidget2')
    assert not dut.tagtypes
    dut.register(test_widget, tags={'k1': ['a']})
    dut.register(test_widget2, tags={'k1': ['b'], 'k2': ['c']})
    assert dut.tagtypes == {'k1', 'k2'}


def test_get_tagvals() -> None:
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    test_widget2 = NamedWidget('TestWidget2')
    dut.register(test_widget, tags={'k1': ['a']})
    dut.register(test_widget2, tags={'k1': ['b'], 'k2': ['c']})
    assert dut.tagvals('k1') == {'a', 'b'}


def test_lookup_multiple_by_mutiple_tag() -> None:
    """Find a list of multiple widgets by matching multiple tags."""
    dut = WidgetCatalog()
    test_widget1 = NamedWidget('TestWidget1')
    test_widget2 = NamedWidget('TestWidget2')
    test_widget3 = NamedWidget('TestWidget3')
    dut.register(test_widget1, 'TestWidget1',
                 tags={'userdef1': ['TestTag'], 'userdef2': ['TestTag']})
    dut.register(test_widget2, 'TestWidget2',
                 tags={'userdef1': ['TestTag'], 'userdef2': ['TestTag']})
    dut.register(test_widget3, 'TestWidget3',
                 tags={'userdef1': ['TestTag'], 'userdef2': ['TestTag2']})
    assert dut.lookup_by_tag_and(userdef1='TestTag', userdef2='TestTag') == {
        'testwidget1': test_widget1,
        'testwidget2': test_widget2,
    }


def test_lookup_multiple_tag_values() -> None:
    """Find a widget that matches at least one tag value out of a list."""
    dut = WidgetCatalog()
    test_widget1 = NamedWidget('TestWidget1')
    test_widget2 = NamedWidget('TestWidget2')
    test_widget3 = NamedWidget('TestWidget3')
    dut.register(test_widget1, tags={'species': 'hamster'})
    dut.register(test_widget2, tags={'species': ['gerbil', 'wolverine']})
    dut.register(test_widget3, tags={'species': 'cuttlefish'})
    assert dut.lookup_by_tag_and(species=('gerbil', 'hamster')) == {
        'testwidget1': test_widget1,
        'testwidget2': test_widget2,
    }


def test_register_with_strings() -> None:
    """Register with strings (instead of lists) in tag values."""
    dut = WidgetCatalog()
    test_widget = NamedWidget('TestWidget')
    assert dut.register(test_widget, tags={'data_type': 'Some Type',
                        'task': 'Some Task'}) == 'testwidget'
    assert dut.lookup_by_tag_and(task='some task') == {'testwidget': test_widget}


def test_load() -> None:
    dut = WidgetCatalog(frob='hello', quux='world')
    dut.load(Path(__file__).parent / 'widgets')

    by_tag = dut.lookup_by_tag_and(some_tag='test_dir')
    assert set(by_tag.keys()) == set({'doodad', 'gizmo'})

    by_name = dut.lookup_by_name('doodad')
    assert hasattr(by_name, 'frob')
    assert hasattr(by_name, 'quux')
    assert by_name.frob == 'hello'
    assert by_name.quux == 'world'


class WidgetWithTags(NamedWidget):
    tags = {'some_tag': ['some_value']}


def test_tag_autoloading():
    dut = WidgetCatalog()
    widget = WidgetWithTags('widget_with_tags')
    dut.register(widget)

    by_tag = dut.lookup_by_tag_and(some_tag='some_value')
    assert set(by_tag.keys()) == set({'widget_with_tags'})


# Test registration with CatalogElementMixin. We're confirming that
# registration picks up _name and _tags.
class WidgetNamedTagged(Widget, CatalogElementMixin):
    _name = 'widget_named_tagged'
    _tags = {
        'tag_key': ['tag_value1', 'tag_value2'],
    }


def test_widget_named_tagged_succeeds():
    dut = WidgetCatalog()
    widget = WidgetNamedTagged()
    dut.register(widget)

    by_name = dut.lookup_by_name(name='widget_named_tagged')
    assert hasattr(by_name, 'name')
    assert by_name.name == widget.name

    by_tag = dut.lookup_by_tag_and(tag_key='tag_value2')
    assert by_tag[widget.name].name == widget.name


def test_lookup_items() -> None:
    """Find a list of several widgets by arbitrary tag."""
    dut = WidgetCatalog()
    test_widgetone = NamedWidget('TestWidget')
    dut.register(obj=test_widgetone, name='TestWidget')
    test_widgettwo = NamedWidget('TestWidgetTwo')
    dut.register(obj=test_widgettwo, name='TestWidgetTwo')

    assert dut.items() == {
        'testwidget': test_widgetone,
        'testwidgettwo': test_widgettwo,
    }.items()
