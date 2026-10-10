from collections import Counter

import pytest

from denarius_model.train import training_rows


@pytest.fixture
def rows(monkeypatch):
    data = [{'category': category, 'id': i} for category in ('A', 'B') for i in range(5)]
    monkeypatch.setattr('denarius_model.train.read_json', lambda path: data)
    return data


def test_subset_is_balanced_deterministic_and_does_not_mutate_source(rows):
    config = {'training': {'seed': 42, 'train_per_category': 2}}
    selected = training_rows(config, 'train')
    assert Counter(r['category'] for r in selected) == {'A': 2, 'B': 2}
    assert selected == training_rows(config, 'train')
    assert len(rows) == 10
    assert training_rows({'training': {'seed': 42}}, 'train') == rows
    assert training_rows(config, 'validation') == rows


@pytest.mark.parametrize('count', [0, -1, True, 1.5, 6])
def test_invalid_subset_size_is_rejected(rows, count):
    with pytest.raises(ValueError):
        training_rows({'training': {'seed': 42, 'train_per_category': count}}, 'train')


def test_final_splits_cannot_be_selected_for_training(rows):
    with pytest.raises(ValueError):
        training_rows({'training': {'seed': 42}}, 'test')
