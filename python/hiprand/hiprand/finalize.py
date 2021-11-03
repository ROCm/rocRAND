"""Finalization with weakrefs

This is designed for avoiding __del__.

http://code.activestate.com/recipes/577242-calling-c-level-finalizers-without-__del__/
by Benjamin Peterson <benjamin@python.org>
"""

from __future__ import print_function

import sys
import traceback
import weakref



class OwnerRef(weakref.ref):
    """A simple weakref.ref subclass, so attributes can be added."""
    pass

def _run_finalizer(ref):
    """Internal weakref callback to run finalizers"""
    del _finalize_refs[id(ref)]
    finalizer = ref.finalizer
    item = ref.item
    try:
        finalizer(item)
    except Exception:
        print("Exception running {}:".format(finalizer), file=sys.stderr)
        traceback.print_exc()

_finalize_refs = {}

def track_for_finalization(owner, item, finalizer):
    """Register an object for finalization.

    ``owner`` is the the object which is responsible for ``item``.
    ``finalizer`` will be called with ``item`` as its only argument when
    ``owner`` is destroyed by the garbage collector.
    """
    ref = OwnerRef(owner, _run_finalizer)
    ref.item = item
    ref.finalizer = finalizer
    _finalize_refs[id(ref)] = ref
