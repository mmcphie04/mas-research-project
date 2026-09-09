import subprocess
import sys

from mad.permute import identity_permutation, make_permutation

SNIPPET = (
    "from mad.permute import make_permutation;print(make_permutation(10, 'q-42', 12345).as_list())"
)


def test_permutation_round_trip():
    perm = make_permutation(10, "q-1", 7)
    for original in range(10):
        assert perm.display_to_original(perm.original_to_display(original)) == original


def test_display_options_follow_the_permutation():
    perm = make_permutation(5, "q-2", 3)
    options = [f"opt{i}" for i in range(5)]
    shown = perm.display_options(options)
    assert shown == [options[i] for i in perm.order]


def test_stable_across_processes_with_different_hash_seeds():
    # builtin hash() is salted per process; a permutation built on it would not reproduce.
    outs = set()
    for seed in ("0", "1", "12345"):
        env = {"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}
        res = subprocess.run(
            [sys.executable, "-c", SNIPPET], capture_output=True, text=True, env=env, check=True
        )
        outs.add(res.stdout.strip())
    assert len(outs) == 1, f"permutation varies with PYTHONHASHSEED: {outs}"


def test_identity():
    assert identity_permutation(4).as_list() == [0, 1, 2, 3]
