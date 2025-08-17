# SSL Certificate Setup with Let's Encrypt

This guide explains how to generate and configure SSL certificates for the live-proxy service using Let's Encrypt.

## Prerequisites

1. A domain name pointing to your server
2. Port 80 and 443 accessible from the internet
3. Certbot installed on your server

## Install Certbot

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install certbot
```

### CentOS/RHEL
```bash
sudo yum install certbot
```

### macOS
```bash
brew install certbot
```

## Generate Certificate

Replace `yourdomain.com` with your actual domain:

### Standalone Mode (temporarily uses port 80)
```bash
sudo certbot certonly --standalone -d yourdomain.com
```

### Webroot Mode (if you have a web server running)
```bash
sudo certbot certonly --webroot -w /var/www/html -d yourdomain.com
```

## Run Your Service with SSL

After certificate generation, start your service with SSL enabled:

```bash
GOOGLE_API_KEY=XXX python proxy.py \
  --cert-file /etc/letsencrypt/live/yourdomain.com/fullchain.pem \
  --key-file /etc/letsencrypt/live/yourdomain.com/privkey.pem \
  --port 443
```

## Certificate Renewal Automation

Let's Encrypt certificates expire every 90 days and need to be renewed automatically.

### Option 1: Cron Job

Create a cron job for automatic renewal:

```bash
# Edit crontab
sudo crontab -e

# Add this line to check for renewal twice daily
0 12,0 * * * certbot renew --quiet && systemctl restart your-service-name
```

### Option 2: Systemd Timer

Create systemd service and timer files for more robust renewal management.

## Important Notes

- **Domain Configuration**: Ensure your domain's DNS A record points to your server's IP address
- **Port Access**: Port 80 must be accessible during certificate generation and renewal
- **Certificate Expiration**: Let's Encrypt certificates expire every 90 days
- **Service Port**: Your service will run on port 443 (HTTPS) when SSL is enabled
- **Certificate Location**: Certificates are stored in `/etc/letsencrypt/live/yourdomain.com/`
- **Renewal Testing**: Test renewal with `sudo certbot renew --dry-run`

## Troubleshooting

### Certificate Generation Fails
- Check that port 80 is open and not blocked by firewall
- Verify domain DNS is pointing to the correct server
- Ensure no other service is using port 80 during generation

### Service Won't Start with SSL
- Verify certificate files exist and are readable
- Check file permissions on certificate files
- Ensure both `--cert-file` and `--key-file` arguments are provided

### Certificate Renewal Issues
- Check cron job logs: `sudo grep CRON /var/log/syslog`
- Test renewal manually: `sudo certbot renew --dry-run`
- Verify port 80 remains accessible for renewal challenges