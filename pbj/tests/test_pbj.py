"""
Unit and regression test for the pbj package.
"""
#python -m pytest 

# Import package, test suite, and other packages as needed
import sys
import pytest
import pbj


def test_pbj_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pbj" in sys.modules


@pytest.fixture
def test():
    print(pbj.PBJ_PATH)
