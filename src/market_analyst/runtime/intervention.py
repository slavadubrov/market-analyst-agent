"""Provider policy stops require operator review, never automatic retry."""


class ProviderIntervention(RuntimeError):
    pass


def raise_if_intervention(error: Exception) -> None:
    body = getattr(error, "body", None)
    if isinstance(error, ProviderIntervention) or "misalignment_policy_violation" in str(body or error):
        raise ProviderIntervention("Provider intervention: stop dispatch and review the run") from error
