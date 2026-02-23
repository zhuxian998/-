import pandas as pd
import numpy as np
import re
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

# 0) 读取数据
file_path = "/content/Mapping6.xlsx"
df = pd.read_excel(file_path)
# 1) 数据预处理/构造 text_all / clean
TEXT_COLS = ["NicknameDT", "Nickname", "Product"]
df["text_all"] = (
    df[TEXT_COLS].fillna("").astype(str).agg(" ".join, axis=1)
      .str.replace(r"\s+", "", regex=True)
)

NEG_REMOVE = r"非(护发素|发膜|去屑|控油|生发|增发|防脱|修复)"
df["text_all_clean"] = df["text_all"].str.replace(NEG_REMOVE, "", regex=True)

df["product_clean"] = (
    df["Product"].fillna("").astype(str)
      .str.replace(r"\s+", "", regex=True)
      .str.replace(NEG_REMOVE, "", regex=True)
)
# =========================
# Nickname 补全：多数票传播 + 模板兜底
# =========================

# 2) 通用函数
def norm_text(s):
    if not isinstance(s, str) or s.strip() == "":
        return ""
    s = s.strip()
    s = re.sub(r"\s+", "", s)         # 去空格
    s = re.sub(r"[／/]", "", s)       # 去掉 / 和 ／
    return s
#主方法的匹配规则
def match_rule(title, must=None, any_=None, exclude=None):
    t = norm_text(title)
    if t == "":
        return False
    if exclude and any(re.search(p, t) for p in exclude):
        return False
    if must and not all(re.search(p, t) for p in must):
        return False
    if any_ and not any(re.search(p, t) for p in any_):
        return False
    return True
#根据描述词，第一形容词为卖点采集方法
def first_hit_label(text, rules, default=np.nan):
    t = norm_text(text)
    if t == "":
        return default

    best_label, best_pos = default, None
    for rule in rules:
        label = rule["label"]
        include = rule.get("include", [])
        exclude = rule.get("exclude", [])

        if exclude and any(re.search(p, t) for p in exclude):
            continue

        pos = None
        for p in include:
            m = re.search(p, t)
            if m:
                pos = m.start() if pos is None else min(pos, m.start())

        if pos is not None and (best_pos is None or pos < best_pos):
            best_label, best_pos = label, pos

    return best_label
# 3) 组键（传播用）
group_cols = ["Brand", "NicknameDT", "Nickname"]
g = df[group_cols].fillna("").astype(str).apply(lambda s: s.str.strip())
key = g.agg("||".join, axis=1)
valid_key = g.apply(lambda col: col != "").all(axis=1)
# A.：三级类目（优先级匹配 + 组内传播）
CAT_TARGET = "三级类目"

def infer_category_priority(text):
    t = norm_text(text)
    if t == "":
        return np.nan

    # 1) 非头发类（优先级最高，直接截断）
    if re.search(r"\+", t) or re.search(r"洗护|旅行装", t):
        return "洗护套装"

    if match_rule(t, any_=[r"沐浴露"]):
        return "沐浴露"
    if match_rule(t, any_=[r"洁面|洗面奶|洁面乳|洗面乳|洁颜|洗颜"]):
        return "洁面"
    if match_rule(t, any_=[r"护理油|身体乳"]):
        return "身体乳"
    if match_rule(t, any_ = ["测试链接"]):
        return "测试链接"
    # 3) 头发类-其它形态
    if match_rule(t, any_=[r"发蜡|发泥|发胶|啫喱|定型|造型|塑型|整理膏"]):
        return "头发造型"
    if match_rule(t, any_=[r"育发|头发护理"]):
        return "头皮护理"
    if match_rule(t, any_=[r"免洗喷雾|干发喷雾"]):
        return "干发喷雾"
    if match_rule(t, any_=[r"假发|头套"]):
        return "假发"
    if match_rule(t, any_=[r"染发|染膏|漂粉|白发"], exclude=[]):
        return "彩染"
    if match_rule(t, any_=[r"磨砂"]):
        return "头皮磨砂膏"
    if match_rule(t, any_=[r"育洗"]):
        return "头皮预洗"

    # 2) 头发类-形态词（优先级高）
    if match_rule(t, any_=[r"发膜|泥膜"]):
        return "发膜"
    if match_rule(t, any_=[r"护发素|养护霜|润发乳|修护膏|精华霜"]):
        return "护发素"
    if match_rule(t, any_=[r"洗发水|洗发露|洗头膏|香波|洗发|洗头"]):
        return "洗发水"

    # 4) 成分/卖点类（优先级最低）——避免“精油沐浴露”被精油抢走
    if match_rule(t, any_=[r"护理油|发油|精油"]):
        return "护发精油"
    if match_rule(t, any_=[r"盲盒|福袋"]):
        return "其他洗护套装"
    if match_rule(t, any_=[r"精华|发油|精油"]):
      if match_rule(t, must=[r"头皮"]):
        return "头皮精油/精华"
      else:
        return "护发精油"
    #if match_rule(t,exclude=[r"头|发"]):
        #return "其他"

    return np.nan

def run_cat_base(df):

    # A1) 基础补全
    cat_pred = df["text_all_clean"].apply(infer_category_priority)
    cat_base_mask = df[CAT_TARGET].isna() & cat_pred.notna()
    df.loc[cat_base_mask, CAT_TARGET] = cat_pred[cat_base_mask]
    print("三级类目 基础补全:", int(cat_base_mask.sum()))

    # A2) 组内传播
    cat_group_nunique = df[CAT_TARGET].fillna("").groupby(key).transform(
        lambda s: (s[s != ""].nunique() if (s != "").any() else 0)
    )
    cat_group_unique = (cat_group_nunique == 1)
    cat_group_label = df[CAT_TARGET].groupby(key).transform(
        lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan
    )
    cat_prop_mask = df[CAT_TARGET].isna() & valid_key & cat_group_unique & cat_group_label.notna()
    df.loc[cat_prop_mask, CAT_TARGET] = cat_group_label[cat_prop_mask]
    print("三级类目 组内传播:", int(cat_prop_mask.sum()))

    return df

# B. Main Function（first-hit + 组内传播 + 兜底）
TARGET = "Main Function"

RULES = [
    {
        "label": "Anti-hair loss 防脱 (国妆特字）",
        "include": [r"抗脱", r"防脱",r"密发",r"育发",r"生发",r"增发",r"固发",r"脱发","何首乌"],
        "exclude": [r"非防脱", r"不是防脱", r"不防脱"]
    },
    {
        "label": "Anti-dandruff 去屑",
        "include": [r"去屑", r"硫化硒", r"头皮屑", r"祛屑",r"头屑"],
        "exclude": [r"非去屑", r"不是去屑", r"不去屑"]
    },
    {
        "label": "Oil control 控油",
        "include": [r"控油", r"去油", r"油头", r"祛油",r"驱油"],
        "exclude": [r"非控油", r"不是控油", r"不控油",r"非去油"]
    },
    {
        "label": "Color lock 锁色",
        "include": [r"固色",r"护色",r"锁色",r"炫亮",r"修护掉色",r"去黄"],
        "exclude": [r"非固色",r"非护色",r"非锁色"]
    },
    {
        "label": "Fluffing & Volumizing 蓬松",
        "include": [r"蓬松", r"高颅顶", r"扁塌"],
        "exclude": []
    },
    {
        "label": "Anti-breakage 强韧防断",
        "include": [r"强韧",r"赋活",r"丰盈",r"发根",r"防断",r"生姜"],
        "exclude": []
    },
    {
        "label": "Fiber Repair 发丝强韧修复",
        "include": [r"修护", r"修复", r"受损", r"焗油", r"滋养", r"干枯","滋润",r"护发",r"发芯",r"烫染受损"],
        "exclude": []
    },
    {
        "label": "Nourish & Hydration 柔顺保湿",
        "include": [r"丝质", r"垂顺", r"顺滑", r"柔顺",r"暗哑",r"丝滑",r"毛躁",r"亮泽"],
        "exclude": []
    },
    {
        "label": "Scalp care 头皮护理",
        "include": [r"止痒", r"舒缓", r"除螨", r"止痒炎",r"敏感",r"舒敏",r"补水",r"毛囊",r"保湿",r"玻尿酸","肌底"],
        "exclude": []
    },
    {
        "label":"Hair dyeing and perming 染发烫发",
        "include": [r"染发",r"烫发",r"染膏"],
        "exclude": []
    },
    {
        "label":"Wig 假发",
        "include": [r"假发",r"头套"],
        "exclude": []
    },
    {
        "label":"香氛留香",
        "include":[r"留香",r"香氛",r"香水",r"香味"]
    },
    {
        "label":"造型",
        "include":[r"定型",r"造型",r"塑型",r"干发",r"发胶",r"发泥",r"发蜡",r"啫喱"]
    }

]
ALL_PAT = "(" + "|".join([p for r in RULES for p in r["include"]]) + ")"
def run_mf_base(df):
    TARGET = "Main Function"
    OTHER  = "Other products 其他产品"

    # 非头发拦截
    non_hair_mask = df["三级类目"].isin(["沐浴露", "洁面", "身体乳", "测试链接", "其他"])

    # B1) 基础补全 first-hit（非头发不打功效）
    pred = df["text_all_clean"].apply(lambda x: first_hit_label(x, RULES, default=np.nan))
    pred = pred.mask(non_hair_mask, np.nan)
    base_mask = df[TARGET].isna() & pred.notna()
    df.loc[base_mask, TARGET] = pred[base_mask]
    print("Main Function 基础补全:", int(base_mask.sum()))

    # B2) 组内传播：text_all_clean 不含任何关键词才传播
    text_has_kw = df["text_all_clean"].str.contains(ALL_PAT, regex=True, na=False)
    group_label_nunique = df[TARGET].fillna("").groupby(key).transform(
        lambda s: (s[s != ""].nunique() if (s != "").any() else 0)
    )
    group_unique = (group_label_nunique == 1)
    group_label = df[TARGET].groupby(key).transform(
        lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan
    )
    prop_mask = (
        df[TARGET].isna() &
        (~text_has_kw) &
        valid_key &
        group_unique &
        group_label.notna() &
        (group_label != OTHER)
    )
    df.loc[prop_mask, TARGET] = group_label[prop_mask]
    print("Main Function 组内传播:", int(prop_mask.sum()))

    # B3) 纠偏：非头发 -> Other
    df.loc[non_hair_mask, TARGET] = OTHER

    # B4) 兜底：仍为空 + 不含发/头 + 且三级类目也为空 -> Other
    other_mask = (
        df[TARGET].isna()
        & (~df["text_all_clean"].fillna("").str.contains(r"(发|头)", regex=True, na=False))
        & (df["三级类目"].isna() | (df["三级类目"].astype(str).str.strip() == ""))
    )
    df.loc[other_mask, TARGET] = OTHER
    print("Main Function 兜底Other:", int(other_mask.sum()))

    return df


#自定义规则方法实现
def apply_post_fix(df, rules, reason_col="post_fix_reason", verbose=True):
    """
    rules: list[dict]
      每条规则格式：
      {
        "name": "规则名",
        "mask": lambda df: 一个布尔mask,
        "set": {"三级类目": "xxx"} 或 {"Main Function": "xxx"} 或多个列一起改
      }
    """
    if reason_col not in df.columns:
        df[reason_col] = ""

    total = 0
    for r in rules:
        m = r["mask"](df)
        n = int(m.sum())
        if n == 0:
            if verbose:
                print(f"[post_fix] {r['name']}: 0")
            continue

        # 执行赋值
        for col, val in r["set"].items():
            df.loc[m, col] = val

        df.loc[m, reason_col] = (df.loc[m, reason_col] + ";" + r["name"]).str.strip(";")
        total += n
        if verbose:
            print(f"[post_fix] {r['name']}: {n}")

    if verbose:
        print(f"[post_fix] total changed rows (may overlap across rules): {total}")
    return df
CAT_COL = "三级类目"
MF_COL  = "Main Function"
TEXT    = "text_all_clean"

POST_RULES = [
    {
        "name": "蓬松+喷雾=>头发造型",
        "mask": lambda d: (
            d[MF_COL].fillna("").str.contains(r"蓬松", regex=True, na=False)
            & d[TEXT].fillna("").str.contains(r"(喷雾)", regex=True, na=False)
            & ~d[CAT_COL].fillna("").isin(["沐浴露", "洁面", "身体乳", "身体护理", "测试链接"])
        ),
        "set": {CAT_COL: "头发造型"}
    },
    {
        "name": "造型类目但Main空=>造型",
        "mask": lambda d: (
            d[CAT_COL].fillna("").isin(["头发造型", "干发喷雾"])
            & (d[MF_COL].isna() | (d[MF_COL].fillna("").astype(str).str.strip() == ""))
        ),
        "set": {MF_COL: "造型"}
    },
    {
        "name": "Main是其他但类目空=>其他",
        "mask": lambda d: (
            d[MF_COL].fillna("").eq("Other products 其他产品")
            & (d[CAT_COL].isna() | (d[CAT_COL].fillna("").astype(str).str.strip() == ""))
        ),
        "set": {CAT_COL: "其他"}
    },
]
df = run_cat_base(df)
df = run_mf_base(df)
df = apply_post_fix(df,POST_RULES)

# ---------------------------
# 0) 配置
# ---------------------------
#MODEL_NAME = "hfl/chinese-roberta-wwm-ext"   # BERT/RoBERTa 中文强烈推荐
MODEL_NAME = "bert-base-chinese"

CAT_THR = 0.55   # 三级类目低置信度阈值（<阈值不写回）
MF_THR  = 0.55   # Main Function 低置信度阈值
RARE_MIN = 3     # 小样本类合并到“其他”的最小样本数
MAX_LEN = 64
EPOCHS = 2
BS_TRAIN = 16
BS_EVAL  = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def _clean_label_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()
    return s

def _merge_rare_labels(y: pd.Series, rare_min=3, other_name="其他") -> pd.Series:
    vc = y.value_counts()
    rare = vc[vc < rare_min].index
    return y.where(~y.isin(rare), other=other_name)

def _make_label_maps(labels: pd.Series):
    uniq = sorted(labels.unique().tolist())
    label2id = {l:i for i,l in enumerate(uniq)}
    id2label = {i:l for l,i in label2id.items()}
    return label2id, id2label

def _build_hf_dataset(texts: pd.Series, labels: pd.Series, label2id: dict):
    df_tmp = pd.DataFrame({"text": texts.astype(str), "labels": labels.map(label2id).astype(int)})
    ds = Dataset.from_pandas(df_tmp)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    ds = ds.map(tok_fn, batched=True)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    return ds

def train_text_clf(texts: pd.Series, y: pd.Series, out_dir: str):
    """
    训练一个BERT分类器并保存到 out_dir
    """
    y = _clean_label_series(y)
    y = _merge_rare_labels(y, RARE_MIN, "其他")

    label2id, id2label = _make_label_maps(y)
    ds = _build_hf_dataset(texts, y, label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=BS_TRAIN,
        per_device_eval_batch_size=BS_EVAL,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
    )
    trainer.train()

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir

@torch.inference_mode()
def predict_missing_with_bert(df, target_col, input_texts: pd.Series, model_dir: str, conf_col: str, thr: float):
    """
    仅对 target_col 为空的行预测；低于thr不写回（保持空）
    """
    # 找“需要预测”的行：target空 且 text不空
    target_s = _clean_label_series(df[target_col])
    text_s = input_texts.fillna("").astype(str).str.strip()

    miss = (target_s == "") & (text_s != "")
    idx = df.index[miss]
    if len(idx) == 0:
        print(f"{target_col}: 没有需要BERT补全的行")
        return df

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()

    # 分batch推理
    batch_size = 64 if DEVICE == "cuda" else 16
    preds = []
    confs = []

    for i in range(0, len(idx), batch_size):
        b_idx = idx[i:i+batch_size]
        b_text = text_s.loc[b_idx].tolist()

        enc = tokenizer(b_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(DEVICE)
        logits = model(**enc).logits
        prob = torch.softmax(logits, dim=-1)

        conf, pred_id = torch.max(prob, dim=-1)
        pred_lab = [model.config.id2label[int(x)] for x in pred_id.cpu().numpy()]

        preds.extend(pred_lab)
        confs.extend(conf.cpu().numpy().tolist())

    # 写回：先写预测与置信度，再把低置信度的清空
    df.loc[idx, target_col] = preds
    df.loc[idx, conf_col] = confs

    low = df.loc[idx, conf_col] < thr
    low_cnt = int(low.sum())
    if low_cnt > 0:
        df.loc[idx[low.values], target_col] = np.nan

    print(f"{target_col} BERT补全: {len(idx)}；低置信度(<{thr})清空: {low_cnt}")
    return df
# ---------------------------
# 1) 先训练 Main Function
# ---------------------------
mf_train_mask = (_clean_label_series(df[MF_COL]) != "") & (_clean_label_series(df[TEXT]) != "")
mf_texts = df.loc[mf_train_mask, TEXT].astype(str)
mf_y     = df.loc[mf_train_mask, MF_COL].astype(str)

mf_model_dir = "/content/mf_bert_model"
print("开始训练Main Function BERT，样本数:", len(mf_texts))
train_text_clf(mf_texts, mf_y, mf_model_dir)

# 只补空白 Main Function
df = predict_missing_with_bert(
    df=df,
    target_col=MF_COL,
    input_texts=df[TEXT],
    model_dir=mf_model_dir,
    conf_col=MF_COL + "_bert_conf",
    thr=MF_THR
)

# ---------------------------
# 2) 再训练 三级类目
# ---------------------------
cat_train_mask = (_clean_label_series(df[CAT_COL]) != "") & (_clean_label_series(df[TEXT]) != "")
cat_texts = df.loc[cat_train_mask, TEXT].astype(str)
cat_y     = df.loc[cat_train_mask, CAT_COL].astype(str)

cat_model_dir = "/content/cat_bert_model"
print("开始训练三级类目BERT，样本数:", len(cat_texts))
train_text_clf(cat_texts, cat_y, cat_model_dir)

# 只补空白 三级类目
df = predict_missing_with_bert(
    df=df,
    target_col=CAT_COL,
    input_texts=df[TEXT],
    model_dir=cat_model_dir,
    conf_col=CAT_COL + "_bert_conf",
    thr=CAT_THR
)

print("BERT补全完成")
def remove_square_bracket_content(text: str) -> str:
    cleaned_text = re.sub(r"【.*?】", "", text)
    return cleaned_text

def extract_core_label(main_func: str):
    if not isinstance(main_func, str):
        return ""
    chinese_part = re.findall(r"[\u4e00-\u9fff]+", main_func)
    if chinese_part:
        return chinese_part[0]
    return ""
def split_brand_anchors(brand: str):
    """
    Brand: 'CLEAR/清扬' -> (['CLEAR'], ['清扬'])
          'Aveda/艾凡达' -> (['Aveda'], ['艾凡达'])
          '999' -> (['999'], [])
          'MY.ORGANICS' -> (['MY.ORGANICS'], [])
    """
    b = brand if isinstance(brand, str) else ""
    b = b.strip()
    if b == "":
        return [], []
    parts = [p.strip() for p in re.split(r"[／/|]", b) if p.strip()]
    cn = [p for p in parts if re.search(r"[\u4e00-\u9fff]", p)]
    en = [p for p in parts if not re.search(r"[\u4e00-\u9fff]", p)]
    if not cn:
        cn = []

    return en, cn

# 三级类目 -> 在 Product 中可作为“类目锚”的同义词
CAT_ANCHORS = {
    "洗发水": [r"洗发水", r"洗发露", r"洗头膏", r"香波", r"洗发", r"洗头"],
    "护发素": [r"护发素", r"养护霜", r"润发乳", r"修护膏", r"精华霜"],
    "发膜":   [r"发膜", r"泥膜"],
    "彩染":   [r"染发", r"染膏", r"染发膏", r"漂粉", r"双氧乳", r"染膏用", r"白发"],
    "干发喷雾": [r"干发喷雾", r"免洗喷雾", r"干洗喷雾", r"干洗剂"],
    "头发造型": [r"发蜡", r"发泥", r"发胶", r"啫喱", r"定型", r"造型", r"塑型", r"整理膏", r"直发膏", r"摩丝"],
    "假发": [r"假发", r"头套"],
    "头皮精油/精华": [r"头皮精华", r"头皮精华液", r"头皮精华油", r"头皮精华喷雾", r"头皮精华露", r"头皮精华液", r"头皮精华安瓶", r"头皮精华", r"头皮护理精华", r"精华液", r"精华"],
    "护发精油": [r"护发精油", r"发油", r"护理油", r"精油"],
    "沐浴露": [r"沐浴露", r"沐浴乳", r"沐浴液"],
    "洁面": [r"洁面", r"洗面奶", r"洁面乳", r"洗面乳", r"洁颜", r"洗颜"],
    "身体乳": [r"身体乳", r"润肤乳", r"身体霜", r"润肤霜"],
    "洗护套装": [r"洗护套装", r"套装", r"洗护", r"\+"],
}

def find_last_brand_anchor_pos(product: str, anchors: list[str]):
    hits = []
    for a in anchors:
        if not a:
            continue
        
        for m in re.finditer(re.escape(a), product):
            hits.append((m.start(), m.end(), a))
    if not hits:
        return None
    return sorted(hits, key=lambda x: (x[0], x[1]))[-1]  

def find_first_cat_anchor_after(product: str, cat_patterns: list[str], start_pos: int):
    best = None
    for pat in cat_patterns:
        m = re.search(pat, product[start_pos:])
        if not m:
            continue
        s = start_pos + m.start()
        e = start_pos + m.end()
        if best is None or s < best[0]:
            best = (s, e, pat)
    return best

def extract_nickname(product: str, brand: str, cat: str,main_func: str = ""):
    """
    核心：用【品牌锚】+【类目锚】截取 product 中间段，生成 nickname（不额外拼 brand）
    """
    p = norm_text(product)
    p = remove_square_bracket_content(p)
    en, cn = split_brand_anchors(brand)
    if p == "":
        # 处理空产品名的情况，直接用品牌 + 类目补全
        if cn:
            return f"{''.join(cn)}{extract_core_label(main_func)}{cat}"
        else:
            return f"{''.join(en)}{extract_core_label(main_func)}{cat}"
    
    
    anchors = en + cn

    brand_hit = find_last_brand_anchor_pos(p, anchors)
    if brand_hit:
        b_start, b_end, b_anchor = brand_hit
        cut_start = b_start  
    else:
        cut_start = 0  
        b_end = 0

    cat_patterns = CAT_ANCHORS.get(cat, [])
    cat_hit = find_first_cat_anchor_after(p, cat_patterns, cut_start)

    if not cat_hit:
      return p

    c_start, c_end, used_pat = cat_hit

    # 截取：从品牌锚开始到“类目锚结束”
    if b_end == 0:
      if main_func == "Wig 假发":
        return f"{''.join(cn)}{cat}"
      if cn:
            return f"{''.join(cn)}{extract_core_label(main_func)}{cat}"
      else:
            return f"{''.join(en)}{extract_core_label(main_func)}{cat}"



    nick = p[b_end:c_start]

    # 清洗常见分隔符（
    nick = re.sub(r"[-_—–·•:：]+", "", nick)
    nick = re.sub(r"[()（）\[\]【】]", "", nick)
    nick = nick.strip()
    if not nick:
      nick = extract_core_label(main_func)
    nick = p[b_start:b_end] + nick + p[c_start:c_end]
  
  

    return nick
df["Nickname"] = df.apply(lambda r: extract_nickname(r["Product"], r["Brand"], r["三级类目"], r["Main Function"]), axis=1)
print("Nickname 补全完成")
# 7) 保存
out = "/content/Mapping6_filled.xlsx"
df_out = df.drop(columns=["text_all", "text_all_clean", "product_clean"], errors="ignore")
df_out.to_excel(out, index=False)
out
