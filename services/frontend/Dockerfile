# services/frontend/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

# Installation des dépendances
COPY package*.json ./
RUN npm install --production

# Copie du code source
COPY public/ ./public/
COPY src/ ./src/

# Build de l'application
RUN npm run build

# Image de production avec Nginx
FROM nginx:alpine

# Copie de la configuration nginx
COPY nginx.conf /etc/nginx/nginx.conf

# Copie de l'application buildée
COPY --from=builder /app/build /usr/share/nginx/html

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]