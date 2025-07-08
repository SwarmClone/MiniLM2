from . import config
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from tqdm import tqdm
import csv

convert_table = {'⼀': '一', '⼁': '丨', '⼂': '丶', '⼃': '丿', '⼄': '乙', '⼅': '亅', '⼆': '二', '⼇': '亠', '⼈': '人', '⼉': '儿', '⼊': '入', '⼋': '八', '⼌': '冂', '⼍': '冖', '⼏': '几', '⼐': '凵', '⼑': '刀', '⼒': '力', '⼓': '勹', '⼗': '十', '⼘': '卜', '⼙': '卩', '⼚': '厂', '⼛': '厶', '⼜': '又', '⼝': '口', '⼞': '囗', '⼟': '土', '⼠': '士', '⼡': '夂', '⼢': '夊', '⼣': '夕', '⼤': '大', '⼥': '女', '⼦': '子', '⼧': '宀', '⼨': '寸', '⼩': '小', '⼪': '尢', '⼫': '尸', '⼬': '屮', '⼭': '山', '⼮': '巛', '⼯': '工', '⼰': '己', '⼱': '巾', '⼲': '干', '⼳': '幺', '⼴': '广', '⼵': '廴', '⼶': '廾', '⼷': '弋', '⼸': '弓', '⼹': '彐', '⼺': '彡', '⼻': '彳', '⼼': '心', '⼽': '戈', '⼾': '戶', '⼿': '手', '⽀': '支', '⽁': '攴', '⽂': '文', '⽃': '斗', '⽄': '斤', '⽅': '方', '⽆': '无', '⽇': '日', '⽈': '曰', '⽉': '月', '⽊': '木', '⽋': '欠', '⽌': '止', '⽍': '歹', '⽎': '殳', '⽏': '毋', '⽐': '比', '⽑': '毛', '⽒': '氏', '⽓': '气', '⽔': '水', '⽕': '火', '⽖': '爪', '⽗': '父', '⽘': '爻', '⽙': '爿', '⽚': '片', '⽛': '牙', '⽜': '牛', '⽝': '犬', '⽞': '玄', '⽟': '玉', '⽠': '瓜', '⽡': '瓦', '⽢': '甘', '⽣': '生', '⽤': '用', '⽥': '田', '⽦': '疋', '⽧': '疒', '⽨': '癶', '⽩': '白', '⽪': '皮', '⽫': '皿', '⽬': '目', '⽭': '矛', '⽮': '矢', '⽯': '石', '⽰': '示', '⽱': '禸', '⽲': '禾', '⽳': '穴', '⽴': '立', '⽵': '竹', '⽶': '米', '⽷': '糸', '⽸': '缶', '⽹': '网', '⽺': '羊', '⽻': '羽', '⽼': '老', '⽽': '而', '⽾': '耒', '⽿': '耳', '⾀': '聿', '⾁': '肉', '⾂': '臣', '⾃': '自', '⾄': '至', '⾅': '臼', '⾆': '舌', '⾇': '舛', '⾈': '舟', '⾉': '艮', '⾊': '色', '⾋': '艸', '⾌': '虍', '⾍': '虫', '⾎': '血', '⾏': '行', '⾐': '衣', '⾑': '襾', '⾒': '見', '⾓': '角', '⾔': '言', '⾕': '谷', '⾖': '豆', '⾗': '豕', '⾘': '豸', '⾙': '貝', '⾚': '赤', '⾛': '走', '⾜': '足', '⾝': '身', '⾞': '車', '⾟': '辛', '⾠': '辰', '⾡': '辵', '⾢': '邑', '⾣': '酉', '⾤': '采', '⾥': '里', '⾦': '金', '⾧': '長', '⾨': '門', '⾩': '阜', '⾪': '隶', '⾫': '隹', '⾬': '雨', '⾭': '青', '⾮': '非', '⾯': '面', '⾰': '革', '⾱': '韋', '⾲': '韭', '⾳': '音', '⾴': '頁', '⾵': '風', '⾶': '飛', '⾷': '食', '⾸': '首', '⾹': '香', '⾺': '馬', '⾻': '骨', '⾼': '髙', '⾽': '髟', '⾾': '鬥', '⾿': '鬯', '⿀': '鬲', '⿁': '鬼', '⿂': '魚', '⿃': '鳥', '⿄': '鹵', '⿅': '鹿', '⿆': '麥', '⿇': '麻', '⿈': '黃', '⿉': '黍', '⿊': '黑', '⿋': '黹', '⿌': '黽', '⿍': '鼎', '⿎': '鼓', '⿏': '鼠', '⿐': '鼻', '⿑': '齊', '⿒': '齒', '⿓': '龍', '⿔': '龜', '⿕': '龠'}

convert = lambda text: "".join([convert_table.get(c, c) for c in text])

def preprocess_csvsft(text_path: str, bin_path: str, mask_path: str, tokenizer: PreTrainedTokenizer, max_length: int):
    with open(text_path, encoding="utf-8") as csvfile, open(bin_path, "wb") as f_bin, open(mask_path, "wb") as f_mask:
        reader = csv.reader(csvfile)
        for row in tqdm(reader):
            _q, _a, q, a = row
            q = convert(q).replace("\n", " ")
            a = convert(a).replace("\n", " ")
            context = [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
            d = tokenizer.apply_chat_template(context, return_assistant_tokens_mask=True, return_dict=True)
            ids, masks = d["input_ids"], d["assistant_masks"]
            padding_size = (max_length + 1) - len(ids) % (max_length + 1)
            f_bin.write(np.array(ids + [config.SPECIAL_TOKENS["<pad>"]] * padding_size, dtype=np.uint16).tobytes())
            f_mask.write(np.array(masks + [False] * padding_size, dtype=np.bool_).tobytes())

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 6:
        print('Usage: python -m minilm2.utils.preprocess_csvsft <encoder_path> <text_path> <bin_path> <mask_path> <max_length>')
        exit(1)
    encoder_path = sys.argv[1]
    text_path = sys.argv[2]
    bin_path = sys.argv[3]
    mask_path = sys.argv[4]
    max_length = int(sys.argv[5])
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, trust_remote_code=True)
    preprocess_csvsft(text_path, bin_path, mask_path, tokenizer, max_length)
