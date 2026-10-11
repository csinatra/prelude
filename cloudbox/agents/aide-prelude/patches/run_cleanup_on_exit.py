"""Shut the REPL child down when aideml's step loop exits by exception.

aideml's `run()` calls `interpreter.cleanup_session()` only after the step loop
finishes normally. If a step raises, the exception unwinds past it, and
multiprocessing's atexit handler then joins the still-running REPL child —
which, having finished its node, waits forever for the next one. The process
neither progresses nor exits, holding the GPU idle until the wall-clock
`timeout` kills it, hours later.

Hit twice: an Anthropic credit exhaustion (2026-09-08) and a malformed review
tool call (2026-10-10, ~3h of idle box). A try/finally around the loop makes any
crash end the run promptly, so it is graded on its best node instead of idling.
No effect on a run that does not crash; identical for every condition.
"""

import aide.run as m

OLD_LOOP = """\
    while global_step < cfg.agent.steps:
        agent.step(exec_callback=exec_callback)
        # on the last step, print the tree
        if global_step == cfg.agent.steps - 1:
            logger.info(journal_to_string_tree(journal))
        save_run(cfg, journal)
        global_step = len(journal)
    interpreter.cleanup_session()
"""

NEW_LOOP = """\
    # PRELUDE: always release the REPL child (patches/run_cleanup_on_exit.py)
    try:
        while global_step < cfg.agent.steps:
            agent.step(exec_callback=exec_callback)
            # on the last step, print the tree
            if global_step == cfg.agent.steps - 1:
                logger.info(journal_to_string_tree(journal))
            save_run(cfg, journal)
            global_step = len(journal)
    finally:
        interpreter.cleanup_session()
"""


def main() -> None:
    src = open(m.__file__).read()
    assert src.count(OLD_LOOP) == 1, "aideml run() step loop changed — re-check this patch"
    open(m.__file__, "w").write(src.replace(OLD_LOOP, NEW_LOOP))
    print("run_cleanup_on_exit: applied")


if __name__ == "__main__":
    main()
