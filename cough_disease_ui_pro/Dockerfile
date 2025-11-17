FROM node:20-bullseye

WORKDIR /app

COPY package.json ./
COPY tsconfig.json ./
COPY babel.config.js ./

RUN npm install -g npm@latest && npm install

COPY . .

ENV EXPO_DEVTOOLS_LISTEN_ADDRESS=0.0.0.0 \
    EXPO_NO_DOTENV=1 \
    EXPO_USE_DEV_SERVER=true \
    EXPO_WEB_PORT=3000 \
    EXPO_PUBLIC_API_URL=http://localhost:8000/v1/infer \
    CI=1 \
    HOST=0.0.0.0

EXPOSE 3000

CMD ["npm", "run", "web"]
