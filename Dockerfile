# base-image declaration & credentials
FROM python:3.6-slim

# Show Python version on build step
RUN python -V

# Build application
ARG APP_DIR=/app
WORKDIR ${APP_DIR}
ADD requirements.txt .
RUN pip --disable-pip-version-check install -r requirements.txt
COPY . ${APP_DIR}
RUN pip --disable-pip-version-check install -e .
RUN chmod -R a+w ${APP_DIR}
ENTRYPOINT ["fraud_detection"]
