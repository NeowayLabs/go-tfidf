FROM alpine:3.9 as base

RUN apk --no-cache update && \
    apk --no-cache add ca-certificates tzdata && \
    rm -rf /var/cache/apk/*

RUN adduser -D -g '' appuser

COPY ./cmd/tf-idf/tf-idf /app/tf-idf

FROM scratch

COPY --from=base /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=base /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=base /etc/passwd /etc/passwd
COPY --from=base /etc/group /etc/group
COPY --from=base /app/tf-idf /app/tf-idf

# Use an unprivileged user.
USER appuser

ENTRYPOINT ["/app/tf-idf"]
