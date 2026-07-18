"""
Unit and regression test for the pbj package.
"""

# python -m pytest
# tests should start with "test_"!
# Import package, test suite, and other packages as needed
import sys
import pytest
import pbj
from .test_single import test_single
from .test_multiple import test_multiple
from .test_formulations import test_formulations


def test_pbj_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pbj" in sys.modules


@pytest.fixture(autouse=True)
def pbj_path():
    print(pbj.PBJ_PATH)


def test_run_single():
    """Run single solute test."""
    test_single()


def test_run_multiple():
    """Run multiple solutes test."""
    test_multiple()


def test_run_formulations():
    """Run formulations test."""
    test_formulations()
