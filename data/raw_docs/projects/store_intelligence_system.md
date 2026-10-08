# Store Intelligence System

A computer vision retail analytics pipeline built from scratch by Ashutosh. This was
his first project involving Docker, FastAPI, or computer vision — he had no prior
experience with any of these three before starting.

## Purpose

Originally built for the Purplle Tech Challenge 2026, submitted before the June 2026
deadline. It has since been used as the centerpiece project for multiple applications,
including Purplle Tech Challenge 2026, Hitwicket, and the Amazon ML Summer School
statement of purpose.

## Technical Stack

- Object detection: YOLOv8n
- Object tracking: ByteTrack
- Re-identification: colour histogram-based Re-ID
- Backend API: FastAPI with 9 endpoints
- Database: SQLite
- Real-time updates: Server-Sent Events (SSE) dashboard
- Narrative generation: Claude API used to generate human-readable summaries of store
  activity
- Deployment: Docker Compose
- Testing: 33 pytest test cases

## Known Limitations

The live dashboard currently uses simulated events for demo purposes rather than live
CCTV footage — this is an important caveat to raise proactively in interviews rather
than let an interviewer discover it.

## Notable Debugging Story

Ashutosh caught a bug where the video processing pipeline's data went stale, but the
dashboard kept displaying old metric values — for example, showing customers in a full
queue when the store was actually empty. He noticed this by manually comparing the live
video/CCTV footage against what the dashboard was reporting, rather than through an
automated check.

## Ownership

Built solo, end to end.
