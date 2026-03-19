# OpenClaw Web Frontend Spec

This document is a handoff note for the next collaborator who will build the web frontend for `VLM_Grasp_Interactive`.

## 1. Goal

Build a dark, high-contrast web UI that lets a user type a command, send it into the OpenClaw pipeline, and see:

- the parsed intent
- the internal dispatch / routing steps
- the current execution state
- the final result

The UI should feel clean, premium, and operational rather than playful.

## 2. Product Shape

The page is a single-screen control panel with three jobs:

1. Accept natural-language robot commands.
2. Show the internal orchestration trace in a readable way.
3. Provide quick preset commands so the user can test the system in one click.

The frontend should not try to implement robot logic itself. It should call the backend and render traces.

## 3. Recommended Layout

### Top Bar

- Left: logo
- Center: project name
- Right: connection status and environment badge

### Main Body

Use a three-column layout on desktop:

- Left: command input and preset chips
- Center: live scene preview or status canvas
- Right: execution trace and debug output

On mobile, stack vertically in this order:

1. input
2. presets
3. trace
4. preview

## 4. Visual Style

Target style:

- background: very dark charcoal or near-black
- text: bright, clean, high-contrast
- accents: cyan, electric blue, warm amber
- surfaces: layered panels with subtle borders, not flat blocks
- feel: technical, premium, minimal

Avoid:

- purple-heavy defaults
- generic SaaS white dashboards
- soft pastel styling
- overly rounded cartoon UI

Suggested palette:

- background: `#0A0F14`
- panel: `#111A22`
- border: `#223042`
- primary text: `#EAF2FF`
- secondary text: `#9BB0C9`
- accent cyan: `#44D7FF`
- accent amber: `#FFB84D`
- success: `#57E389`
- error: `#FF6B6B`

Typography:

- use a distinctive sans-serif or geometric display face for the header
- keep body text highly legible
- avoid default system fonts as the main visual identity

## 5. Logo Direction

The logo should combine:

- a robot arm silhouette
- a small lobster or crayfish motif

Design intent:

- readable at small sizes
- works as a simple SVG
- can be monochrome or two-tone
- should not look cute or childish

Suggested concept:

- robot arm as the outer stroke
- lobster tail or claw as the inner negative space
- use one sharp, mechanical contour rather than a detailed mascot

## 6. Command Input

The main input should be a large single-line or multiline command box.

Requirements:

- placeholder text should explain what to type
- `Enter` submits
- `Shift+Enter` inserts a newline if multiline is supported
- the input should keep focus by default
- show a clear loading / running state after submit

Suggested placeholder:

```text
Type a command like: place the apple on the plate
```

## 7. Preset Commands

Below the input box, add quick chips for common tests.

Good preset examples:

- `dance`
- `teleop`
- `place the apple on the plate`
- `put the banana next to the duck`
- `put the glass cup on the shelf`
- `drop the apple into the plate`
- `move the knife away from the flower vase`
- `save view`
- `import apple_2`
- `exit`

Include a few fun but harmless commands for demo value.

Recommended additional ideas:

- `spin in place`
- `wave the gripper`
- `move the duck to the center`
- `put the hammer beside the plate`
- `open teleop`
- `show execution trace`

Do not include commands that imply real-world destructive use unless the backend is clearly simulation-only.

## 8. Execution Trace

This is the core differentiator of the UI.

When the user submits a command, the frontend should render the internal process as a live trace, for example:

1. user input received
2. language parse
3. object detection / segmentation
4. task routing
5. grasp target estimation
6. place target estimation
7. IK solve
8. motion execution
9. result

The trace should be shown as a stepper or log stream, with the active step highlighted.

Example trace payload shape:

```json
{
  "type": "pick_place",
  "source": "apple",
  "destination": "plate",
  "relation": "on_top_of",
  "steps": [
    {"name": "parse", "status": "done"},
    {"name": "segment_source", "status": "done"},
    {"name": "segment_destination", "status": "done"},
    {"name": "estimate_grasp", "status": "done"},
    {"name": "estimate_place", "status": "running"},
    {"name": "solve_ik", "status": "pending"},
    {"name": "execute", "status": "pending"}
  ]
}
```

If the backend can stream partial progress, the UI should update in place instead of waiting for a single final response.

## 9. Backend Contract

The frontend should assume a backend API that returns:

- parsed command
- current step
- final status
- optional logs
- optional images / overlays

Suggested endpoints:

- `POST /api/command`
- `GET /api/session/:id`
- `GET /api/session/:id/events`

If there is no streaming backend yet, start with polling or mock data, but keep the UI ready for streaming later.

## 10. Scene Preview

If a scene preview panel is added, it should show:

- current scene screenshot
- optional bounding boxes
- optional segmentation mask overlay
- current selected object or target marker

This panel does not need to be a 3D editor. A live image preview is enough for the first version.

## 11. Interaction States

The UI should clearly distinguish:

- idle
- running
- success
- failure
- interrupted

Each state should have a strong visual treatment:

- idle: calm neutral panel
- running: active cyan glow or progress bar
- success: green success mark
- failure: red error line and message
- interrupted: dimmed / paused state

## 12. Empty States

If no command has been run yet, show a short guidance block:

- what the system does
- a few sample commands
- what the user should expect to see

This should not be a long tutorial. It should just reduce the first-use friction.

## 13. Suggested Tech Approach

Any modern frontend stack is acceptable, but the implementation should prioritize:

- clean state handling
- streaming-friendly rendering
- responsive layout
- easy theming

If using React:

- keep state for command, trace, preview, and session id separated
- keep the command box always ready
- use a lightweight log/trace component

## 14. Acceptance Criteria

The first frontend version is good enough if:

- the user can type a command and submit it
- the UI shows a visible dispatch trace
- preset commands work
- the page looks intentional in dark mode
- the logo is recognizable and not generic
- the page remains usable on a laptop screen

## 15. Notes for Handoff

This frontend is meant to sit on top of the existing OpenClaw pipeline in this repository:

- command routing lives in `openclaw_like`
- grasp / place logic lives in `grasp_process.py`
- scene editing is separate in `scene_layout_editor.py`

The frontend should not duplicate those responsibilities. It should only present them well.

