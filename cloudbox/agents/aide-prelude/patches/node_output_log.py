"""Tee each node's stdout/stderr to $LOGS_DIR/node_output.log as it is written.

aideml runs a node's code in a child process whose sys.stdout/sys.stderr are a
RedirectQueue: every write goes onto a multiprocessing queue that the parent
drains only when the node finishes, into `term_out`. A node can train for hours
with its losses and progress invisible until then — the one artifact a stalled
or slow node never produces is the output that would explain it.

This appends the same text to a file as it is written, with a header per node
(each node runs in a fresh session, so the RedirectQueue is rebuilt per node).
Observability only, identical for every condition: the queue — and so what the
agent sees — is unchanged, and a failed write is swallowed.
"""

import aide.interpreter as m

OLD_CLASS = (
    "class RedirectQueue:\n"
    "    def __init__(self, queue):\n"
    "        self.queue = queue\n"
    "\n"
    "    def write(self, msg):\n"
    "        self.queue.put(msg)\n"
)

NEW_CLASS = (
    "class RedirectQueue:\n"
    "    def __init__(self, queue):\n"
    "        self.queue = queue\n"
    "        # PRELUDE: live tee of node output (patches/node_output_log.py)\n"
    "        try:\n"
    '            path = os.path.join(os.environ.get("LOGS_DIR", "."), "node_output.log")\n'
    '            self._tee = open(path, "a", buffering=1)\n'
    '            self._tee.write(f"\\n===== node started {time.strftime(\'%Y-%m-%d %H:%M:%S\')} =====\\n")\n'
    "        except Exception:\n"
    "            self._tee = None\n"
    "\n"
    "    def write(self, msg):\n"
    "        self.queue.put(msg)\n"
    "        if self._tee is not None:\n"
    "            try:\n"
    "                self._tee.write(msg)\n"
    "                self._tee.flush()\n"
    "            except Exception:\n"
    "                pass\n"
)


def main() -> None:
    src = open(m.__file__).read()
    assert src.count(OLD_CLASS) == 1, "aideml RedirectQueue changed — re-check this patch"
    assert "import time\n" in src and "import os\n" in src, "aideml interpreter imports changed"
    open(m.__file__, "w").write(src.replace(OLD_CLASS, NEW_CLASS))
    print("node_output_log: applied")


if __name__ == "__main__":
    main()
