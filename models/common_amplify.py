# models/common_amplify.py
import random, time
from amplify import solve                # the normal solver

# --- helper ---------------------------------------------------------------

def _is_retryable(err: Exception) -> bool:
    """
    Return True for errors that deserve an automatic retry/back‑off.
    We keep it simple: network time‑outs or the generic Amplify
    'maximum retries exceeded' RuntimeError.
    """
    txt = str(err).lower()

    # Any of these substrings → retry
    retry_keys = (
        "timeout",                # socket read / connect time‑out
        "429", "too many",        # HTTP 429 or quota exceeded
        "maximum retries exceeded",
        "internal server error",  # occasional 500 from the API
        "service unavailable",    # 503
    )
    return any(k in txt for k in retry_keys)


def safe_solve(model, client, *,
               num_solves     = 1,
               max_attempts   = 6,
               base_delay_sec = 1.0,
               max_delay_sec  = 60.0,
               jitter_frac    = 0.2):
    """
    Call `amplify.solve` with exponential back‑off.

    Parameters
    ----------
    model, client         : as usual for solve(...)
    num_solves            : int, forwarded
    max_attempts          : max total tries (1st call + retries)
    base_delay_sec        : initial wait before the first retry
    max_delay_sec         : ceiling for the back‑off
    jitter_frac           : 0‒1, random ± jitter on each sleep

    Returns
    -------
    amplify.Result        : best solution

    Raises
    ------
    Exception             : last non‑retryable or exceeded‑attempts error
    """
    delay = base_delay_sec

    for attempt in range(1, max_attempts + 1):
        try:
            return solve(model, client, num_solves=num_solves)

        except Exception as err:
            # retry only if the error looks transient *and* attempts remain
            if attempt == max_attempts or not _is_retryable(err):
                raise                       # bubble up permanently

            slp = delay * (1 + jitter_frac * (2*random.random() - 1))
            print(f"[safe_solve] attempt {attempt}/{max_attempts} failed "
                  f"({err}).  Sleeping {slp:.1f}s …", flush=True)
            time.sleep(slp)
            delay = min(delay * 2, max_delay_sec)

