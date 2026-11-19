@echo off
rem Reinstall the Pyro node-group assets (Add Modifier > Physics > Pyro Simulation, Add > Pyro)
rem and the viewport's Pyro tab into this Blender, from smoke_assets.py and pyro_tools.py in
rem this folder. Close Blender first. A newer
rem smoke_assets.py dropped into this folder updates the presets and blocks without a new build.
cd /d "%~dp0"
if not exist blender.exe (
  echo Put this file in the folder that holds blender.exe.
  pause
  exit /b 1
)
blender.exe -b --factory-startup --python smoke_assets.py
if errorlevel 1 (
  echo Updating the Pyro assets failed; see the messages above.
) else (
  echo The Pyro assets are up to date. Start Blender as usual.
)
pause
