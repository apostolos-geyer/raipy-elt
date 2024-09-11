from raipy_elt.data_quality.models import (
    ColumnDomainResult,
    EQUAL,
    LEFT_NOT_RIGHT,
    RIGHT_NOT_LEFT,
    BOTH,
)


def test_column_domain_result_equal():
    result = ColumnDomainResult.equal()
    assert result.rel == EQUAL
    assert result.lnotr is None
    assert result.rnotl is None


def test_column_domain_result_left_not_right():
    result = ColumnDomainResult.left_not_right(["missing"])
    assert result.rel == LEFT_NOT_RIGHT
    assert result.lnotr == ["missing"]
    assert result.rnotl is None


def test_column_domain_result_right_not_left():
    result = ColumnDomainResult.right_not_left(["extra"])
    assert result.rel == RIGHT_NOT_LEFT
    assert result.lnotr is None
    assert result.rnotl == ["extra"]


def test_column_domain_result_both():
    result = ColumnDomainResult.both(["left_missing"], ["right_missing"])
    assert result.rel == BOTH
    assert result.lnotr == ["left_missing"]
    assert result.rnotl == ["right_missing"]
