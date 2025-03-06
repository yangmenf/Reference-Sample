import os
from flask import Flask, render_template, request, jsonify
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
import hashlib
from Crypto.Util.Padding import pad, unpad

app = Flask(__name__)

# 加密后的病历文件夹路径
ENCRYPTED_RECORDS_FOLDER = "encrypted_records"
os.makedirs(ENCRYPTED_RECORDS_FOLDER, exist_ok=True)  # 创建文件夹，如果文件夹不存在

# 16字节的AES密钥
KEY = b'1234567890123456'  # 这里设置固定的AES密钥

# 明文加密
def encrypt(plain_text):
    if isinstance(plain_text, str):  # 如果是字符串，先转换为字节串
        plain_text = plain_text.encode()  # 将字符串编码为字节串
    cipher = AES.new(KEY, AES.MODE_CBC)  # 使用CBC模式
    ct_bytes = cipher.encrypt(pad(plain_text, AES.block_size))  # 填充明文到AES块大小
    iv = cipher.iv  # 获取IV
    return iv + ct_bytes  # 将IV和密文拼接在一起


# 解密
def decrypt(encrypted_data):
    iv = encrypted_data[:16]  # 获取IV
    ct = encrypted_data[16:]  # 获取密文
    cipher = AES.new(KEY, AES.MODE_CBC, iv)  # 使用同样的IV和密钥
    pt_bytes = unpad(cipher.decrypt(ct), AES.block_size)  # 解密并去除填充

    try:
        return pt_bytes.decode('utf-8')  # 尝试以utf-8解码
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试使用Windows-1252（ANSI）编码
        try:
            return pt_bytes.decode('windows-1252')  # 使用Windows编码（ANSI）
        except UnicodeDecodeError:
            # 如果两种编码都失败，返回字节串
            return pt_bytes  # 或者直接返回解密后的字节流


# 生成一个16字节（128位）的随机AES密钥
def generate_aes_key():
    return os.urandom(16)  # 生成16字节的AES密钥（128位）

# RSA签名
def rsa_sign(data, private_key):
    key = RSA.import_key(private_key)
    hash_data = SHA256.new(data.encode())
    signer = pkcs1_15.new(key)
    signature = signer.sign(hash_data)
    return signature

# 分段RSA加密
def rsa_encrypt_data(data, public_key):
    key = RSA.import_key(public_key)
    cipher = PKCS1_OAEP.new(key)

    # 分段加密：我们将数据分块加密，每个块的大小不能超过RSA的限制
    max_data_size = key.size_in_bytes() - 42  # 2048位RSA密钥，OAEP填充后，最多能加密key.size_in_bytes() - 42字节
    encrypted_data = b""

    # 分块加密
    for i in range(0, len(data), max_data_size):
        block = data[i:i + max_data_size]
        encrypted_data += cipher.encrypt(block)

    return encrypted_data

# 使用RSA加密AES密钥
def rsa_encrypt_key(aes_key, public_key):
    return rsa_encrypt_data(aes_key, public_key)

# 计算数字摘要（哈希）
def generate_hash(data):
    return hashlib.sha256(data.encode()).hexdigest()

# 主程序逻辑：处理病历加密过程
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取用户输入的数据
        data = request.json
        patient_info = data.get('patient_info')
        doctor_diagnosis = data.get('doctor_diagnosis')
        patient_id = data.get('patient_id')
        doctor_id = data.get('doctor_id')

        # 示例RSA密钥，通常这些应该从配置文件中读取
        patient_public_key = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA8VmgJyyXW/PORN6du44y
QE1495+oUruvJRAeaEn+/YEeMN45pwsaP7O20YSZzOLw5Ku8maVUidG5HKOcqKje
p+8hezvuXbDf0S6Z4Uz1uBbI2cmkM1R8Cz4UyEClnfL51cy9F7FAhCP93Htt6IuB
qybEToMVtiwPjHT8/1dzrrqjnlAQAOaXUEzeOC0s/7TYuXnr8MvQw9rxiT+nSyRX
ZjRvTqFc/xr9MTYP6oyC/b9AoLHUZBYVzQvUSV+dIKjnaPNj+cATvXvOYcHbjz5G
RoHwiUsqqde4Og7FxcOg4oNnTI2lQBPAaVjw9PZfqBfxGGxcsvg22lyG73y+LIBK
+QIDAQAB
-----END PUBLIC KEY-----"""  # 患者的RSA公钥

        # 生成医生的RSA密钥对
        doctor_key = RSA.generate(2048)
        doctor_private_key = doctor_key.export_key()
        doctor_public_key = doctor_key.publickey().export_key()

        # 生成AES密钥
        aes_key = generate_aes_key()

        # 加密数据
        encrypted_data, aes_key_encrypted = process_medical_record(patient_info, doctor_diagnosis, aes_key,
                                                                   patient_public_key, doctor_private_key)

        # 保存加密后的病历数据和AES密钥
        save_encrypted_record(patient_id, encrypted_data)
        save_aes_key(patient_id, aes_key_encrypted)

        # 返回加密后的数据和加密的AES密钥（以十六进制返回）
        return jsonify({
            'encrypted_data': encrypted_data.hex(),  # 以十六进制返回加密数据
            'aes_key': aes_key_encrypted.hex(),  # 以十六进制返回加密的AES密钥
        })

    return render_template('index.html')

# 病历加密过程
def process_medical_record(patient_info, doctor_diagnosis, aes_key, patient_public_key, doctor_private_key):
    # 1. AES加密患者个人信息（明文1）
    cipher_text1 = encrypt(patient_info)

    # 2. 医生解密明文1（AES解密）
    decrypted_text1 = decrypt(cipher_text1)

    # 3. 在明文1的基础上添加诊断信息（明文2）
    plain_text2 = decrypted_text1 + doctor_diagnosis

    # 4. 对明文2进行哈希，并附加到明文2后
    hash_value = generate_hash(plain_text2)
    plain_text2_with_hash = plain_text2 + "\nHash: " + hash_value

    # 5. 使用医生的RSA私钥对明文2+哈希进行签名
    signature = rsa_sign(plain_text2_with_hash, doctor_private_key)
    signed_data = plain_text2_with_hash + "\nSignature: " + signature.hex()

    # 6. 使用患者的RSA公钥对数字摘要和签名进行加密
    hash_and_signature_data = (hash_value + "\n" + signature.hex()).encode()
    encrypted_hash_and_signature = rsa_encrypt_data(hash_and_signature_data, patient_public_key)

    # 7. 使用AES加密所有内容（明文2 + 数字摘要和签名的密文）
    final_content = plain_text2_with_hash.encode() + b"\nEncryptedHashAndSignature: " + encrypted_hash_and_signature
    encrypted_final_data = encrypt(final_content)  # 使用相同的aes_key

    # 8. 使用RSA加密AES密钥
    encrypted_aes_key = rsa_encrypt_key(aes_key, patient_public_key)

    return encrypted_final_data, encrypted_aes_key

# 保存加密后的病历数据到文件（保存为txt格式）
def save_encrypted_record(patient_id, encrypted_data):
    file_path = os.path.join(ENCRYPTED_RECORDS_FOLDER, f"{patient_id}_encrypted_data.txt")
    with open(file_path, "wb") as file:
        # 只保存加密后的病历数据
        file.write(b"Encrypted Data:\n")
        file.write(encrypted_data)  # 直接保存二进制加密数据

# 保存加密的AES密钥到文件（保存为txt格式）
def save_aes_key(patient_id, aes_key_encrypted):
    file_path = os.path.join(ENCRYPTED_RECORDS_FOLDER, f"{patient_id}_aes_key_encrypted.txt")
    with open(file_path, "wb") as file:
        # 保存加密后的AES密钥
        file.write(b"AES Key Encrypted:\n")
        file.write(aes_key_encrypted)  # 直接保存二进制加密后的AES密钥

# 允许上传加密的病历文件进行解密
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    file = request.files['file']

    try:
        # 固定的AES密钥
        aes_key = b'1234567890123456'  # 这里是你加密时使用的AES密钥

        # 读取上传的文件内容（假设文件是一个加密后的病历）
        file_data = file.read()

        # 解密病历内容
        decrypted_data = decrypt(file_data)

        # 将解密数据转为字符串，或者以十六进制格式返回
        if isinstance(decrypted_data, bytes):
            decrypted_data = decrypted_data.decode('utf-8', errors='ignore')  # 将字节转为字符串
            # 如果解码失败，可以选择返回十六进制字符串
            # decrypted_data = decrypted_data.hex()  # 如果你更喜欢十六进制字符串

        return jsonify({
            'decrypted_data': decrypted_data  # 返回解密后的数据（字符串）
        })

    except ValueError as e:
        return jsonify({'error': '解密失败: ' + str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
