# CapUnlocker provenance

`CapUnlocker.skprx` is the unchanged official GrapheneCt CapUnlocker v1.4 binary.

- Source: https://github.com/GrapheneCt/CapUnlocker
- Release: https://github.com/GrapheneCt/CapUnlocker/releases/tag/v1.4
- Reviewed source revision: `5698ad6d06ab1a694702db0fd6df3927de6b1df5`
- SHA-256: `40df9f495275fd5780966827e69f6a83f6abb0bee5ea608d558a10c06d897da0`
- License: MIT, retained in `licenses/CapUnlocker-LICENSE` and packaged alongside the plugin.

MeteorLLM probes permission to create a fourth-core thread. If unavailable, it
loads this kernel plugin at runtime, then probes again. Failure falls back to
three workers. It does not edit taiHEN boot configuration or set CPU clocks.
The plugin changes system capabilities until reboot; it is not unloaded while
threads using its capabilities may still exist.

Create `ux0:data/VitaDesk/MeteorLLM-08/three-core.txt` and restart to disable
automatic loading and force three workers. Reboot to clear a previously loaded
runtime plugin. This does not override a plugin installed separately at boot.
