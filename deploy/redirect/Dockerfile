# Featherweight redirect app for the OLD Fly URL.
# Replaces the heavy SciNCL app on `siip-armchair-akhil` with a tiny nginx that
# 301-redirects every path (preserving path + query) to the new HF Space.
# Runs on a 256 MB machine with auto-stop -> ~$0/mo.
FROM nginx:alpine

RUN printf 'server {\n\
    listen 8080;\n\
    server_name _;\n\
    return 301 https://dogeyboy19-armchair.hf.space$request_uri;\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
