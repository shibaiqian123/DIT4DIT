# Psi0 test-time RTC attribution

Reference repository: https://github.com/physical-superintelligence-lab/Psi0
Pinned commit: cda3fdfafe47cb6fa75400d013de1bd36e3cea05
Original files: src/psi/models/psi0.py (predict_action_with_rtc_flow, _create_soft_mask), src/psi/deploy/serve_psi0_sonic.py.
The exponential mask and VJP correction are adapted from that version. Original license is reproduced in PSI0_LICENSE.
Differences: original DiT4DiT four-step Euler schedule retained; strict overlap validation rather than silent d/s clamping; scheduling uses client consumption rather than server ticks; no pseudo initial action; 50x66 normalized actions, then split 64+2. No checkpoint training or parameter additions.
A weight-one prefix is guidance, NOT hard freezing of actions. No guarantee of exact prefix equality or physical continuity.
