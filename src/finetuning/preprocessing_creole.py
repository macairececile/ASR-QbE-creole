import re

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\\n\-\_\'\…\[\]]'
a_vowels = '[\à\â]'
e_vowels = '[\ë\ê]'
e_vowels_v3 = '[\ë\ê\è\é]'


def remove_special_characters_v1(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch


def remove_special_characters_v2(batch):
    """Version (a, é, è, e, o, ò, i, u)"""
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    batch["sentence"] = re.sub(a_vowels, 'a', batch["sentence"])
    batch["sentence"] = re.sub(e_vowels, 'è', batch["sentence"])
    batch["sentence"] = re.sub('î', 'i', batch["sentence"])
    batch["sentence"] = re.sub('ù', 'u', batch["sentence"])
    return batch


def remove_special_characters_v3(batch):
    """Version (a, e, i, o, u)"""
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    batch["sentence"] = re.sub(a_vowels, 'a', batch["sentence"])
    batch["sentence"] = re.sub(e_vowels_v3, 'e', batch["sentence"])
    batch["sentence"] = re.sub('î', 'i', batch["sentence"])
    batch["sentence"] = re.sub('ù', 'u', batch["sentence"])
    batch["sentence"] = re.sub('ò', 'o', batch["sentence"])
    return batch
