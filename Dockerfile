# The engine, offline heuristic, and observatory server are pure stdlib, so the
# image is just Python + the source. A real-LLM brain is opt-in and also
# dependency-free (it speaks HTTP via urllib).
FROM python:3.14-slim

# Don't buffer stdout/stderr, so container logs appear immediately.
ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8800

# Run from the source tree (no install): the package has no third-party deps,
# and running in place keeps the data files the server reads at runtime — e.g.
# emergence/web/observatory.html, loaded relative to __file__ — exactly where it
# expects them, without needing package-data wiring.
WORKDIR /app
COPY . .

# Run as a non-root user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8800
CMD ["python", "-m", "emergence.server"]
