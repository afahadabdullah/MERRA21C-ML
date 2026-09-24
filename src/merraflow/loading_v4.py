"""Disposable, exact-value cache for input-only v4 patch proposals."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np


class ProposalCache:
    def __init__(self, root, fingerprint, patch, shape, candidates):
        contract = dict(algorithm='v4-coarse-proposal-1', archive=fingerprint,
                        patch=patch, shape=list(shape))
        key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        self.root = Path(root)/key
        self.candidates = candidates
        self.enabled = True
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.disable(exc)

    def disable(self, exc):
        if self.enabled:
            print(f'Proposal cache unavailable ({exc}); computing proposals normally.', flush=True)
        self.enabled = False

    def load(self, entry_id):
        if not self.enabled:
            return None
        try:
            q = np.load(self.root/f'{entry_id}.npy', allow_pickle=False)
            if (q.shape == (self.candidates,) and q.dtype == np.float64
                    and np.isfinite(q).all() and (q >= 0).all()
                    and np.isclose(q.sum(), 1., rtol=0, atol=1e-12)):
                return q
        except (OSError, ValueError, EOFError):
            pass  # Missing or damaged caches are regenerated from coarse inputs.
        return None

    def save(self, entry_id, q):
        if not self.enabled:
            return
        temporary = None
        try:
            # Workers/ranks may compute the same hour simultaneously. Readers
            # only see complete files; duplicate writers produce identical q.
            with tempfile.NamedTemporaryFile(dir=self.root, prefix='.proposal-',
                                             suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                np.save(stream, q, allow_pickle=False)
            os.replace(temporary, self.root/f'{entry_id}.npy')
        except OSError as exc:
            self.disable(exc)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
