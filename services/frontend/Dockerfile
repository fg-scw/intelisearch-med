FROM nginx:alpine

# Copie de la configuration nginx
COPY nginx.conf /etc/nginx/nginx.conf

# Copie de l'interface web
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]