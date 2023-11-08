dict_tag_byte = {
    'B-Sub': b'\x01\x00',
    'I-Sub': b'\x02\x00',
    'O': b'\x03\x00',
    'B-Pre': b'\x04\x00',
    'I-Pre': b'\x05\x00',
    ',': b'\x06\x00',
    'B-Asp': b'\x07\x00',
    'I-Asp': b'\x08\x00',
    'B-Obj': b'\t\x00',
    'I-Obj': b'\n\x00',
    '.': b'\x0b\x00',
    '+': b'\x0c\x00',
    ':': b'\r\x00',
    '“': b'\x0e\x00',
    '”': b'\x0f\x00',
    '/': b'\x10\x00',
    '(': b'\x11\x00',
    '-': b'\x12\x00',
    ')': b'\x13\x00',
    '"': b'\x14\x00',
    ';': b'\x15\x00',
    '@': b'\x16\x00',
    '!': b'\x17\x00'
}

A = ['\x03', '\x03', '\x03', '\x03', '\x06', '\x01', '\x02']

def check_exit_in_value(input_string, byte_string): 
    input_byte_string = input_string.encode('utf-8')
    if input_byte_string in byte_string:
        return True
    return False

keys_in_A = []
for key, value in dict_tag_byte.items():
    if value in map(bytes, A):
        keys_in_A.append(key)

print(keys_in_A)