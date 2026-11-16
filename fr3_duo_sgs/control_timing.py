"""OmniReset policy timing with unchanged fine Newton contact integration."""
PHYSICS_DT_S = .002
POLICY_DT_S = .1
CONTROL_HZ = 10
CONTROL_DECIMATION = round(POLICY_DT_S / PHYSICS_DT_S)
EPISODE_STEPS = 160
EPISODE_DURATION_S = EPISODE_STEPS * POLICY_DT_S
# Preserve the prior 50Hz collision/success checks and three-check hold gate.
AUDIT_DT_S = .02
AUDIT_DECIMATION = round(AUDIT_DT_S / PHYSICS_DT_S)
SUCCESS_AUDIT_STEPS = 3
assert CONTROL_DECIMATION == 50 and AUDIT_DECIMATION == 10
assert CONTROL_DECIMATION % AUDIT_DECIMATION == 0
