"""Preparação do corpus sintético; nunca lê data/raw ou dados privados."""
from .synthetic import generate_dataset, audit_public_dataset

def prepare():
    stats=generate_dataset()
    audit_public_dataset()
    return stats
