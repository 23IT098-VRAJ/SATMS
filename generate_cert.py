from OpenSSL import crypto
import os

def generate_self_signed_cert(cert_file, key_file):
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(cert_file), exist_ok=True)
    
    # Create a key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 4096)
    
    # Create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().C = "US"
    cert.get_subject().ST = "State"
    cert.get_subject().L = "City"
    cert.get_subject().O = "Organization"
    cert.get_subject().OU = "Organizational Unit"
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(10*365*24*60*60)  # 10 years
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')
    
    # Write certificate and key files
    with open(cert_file, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open(key_file, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    
    print(f"Certificate generated: {cert_file}")
    print(f"Private key generated: {key_file}")

if __name__ == "__main__":
    cert_file = "config/ssl/cert.pem"
    key_file = "config/ssl/key.pem"
    generate_self_signed_cert(cert_file, key_file)